import argparse
import math
import os
import random
import time
from matplotlib import cm, rcParams
from rich.console import Console
from rich.table import Table
from rich import box
from PIL import Image

import cv2
import numpy as np

import torch
import torch.nn.parallel
import torch.utils.data
import visibility
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from datasets.DatasetExtrinsicCalib import DatasetGeneralExtrinsicCalib, DatasetPandasetExtrinsicCalib, \
    DatasetHerculesRadarExtrinsicCalib
from camera_model import CameraModel

from models.get_model import get_model
from quaternion_distances import quaternion_loss

import matplotlib
try:
    matplotlib.use('TkAgg')
except Exception:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
from utils import (downsample_depth, merge_inputs, get_flow_zforward, quat2mat, tvector2mat,
                   quaternion_from_matrix, EndPointError, rotate_forward, quaternion_median,
                   average_quaternions, rotate_back, quaternion_mode, str2bool, overlay_imgs)

rcParams["figure.raise_window"] = False
torch.backends.cudnn.benchmark = True
torch.use_deterministic_algorithms(False)


def infer_hercules_img_shape(dataset):
    if len(dataset) == 0:
        raise RuntimeError("No Hercules samples found for evaluation.")
    first_image = Image.open(dataset.all_files[0]['image_path']).convert('RGB')
    width = max(1, int(round(first_image.width * dataset.image_resize_scale)))
    height = max(1, int(round(first_image.height * dataset.image_resize_scale)))
    return [64 * ((height + 63) // 64), 64 * ((width + 63) // 64)]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCH = 1


def _init_fn(worker_id, seed):
    seed = seed + worker_id + EPOCH * 100
    print(f"Init worker {worker_id} with seed {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def quaternion_distance(q, r):
    dist = quaternion_loss(q.unsqueeze(0), r.unsqueeze(0), q.device)

    dist = 180. * dist.item() / math.pi
    return dist


def prepare_input(cam_params, pc_rotated, real_shape, reflectance, _config, change_frame=False):
    cam_model = CameraModel()
    cam_model.focal_length = cam_params[:2]
    cam_model.principal_point = cam_params[2:]
    uv, depth, _, refl = cam_model.project_pytorch(pc_rotated, real_shape, reflectance, change_frame)
    uv = uv.t().int().contiguous()
    depth_img = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float)
    depth_img += 1000.
    depth_img = visibility.depth_image(uv, depth, depth_img, uv.shape[0], real_shape[1], real_shape[0])
    depth_img[depth_img == 1000.] = 0.

    depth_img_no_occlusion = torch.zeros_like(depth_img, device='cuda')
    depth_img_no_occlusion = visibility.visibility2(depth_img, cam_params, depth_img_no_occlusion,
                                                    depth_img.shape[1], depth_img.shape[0],
                                                    _config['occlusion_threshold'], _config['occlusion_kernel'])

    uv = uv.long()

    # Check valid indexes: as multiple points MIGHT be projected into the same pixel, keep only
    # the points that are actually projected
    indexes = depth_img_no_occlusion[uv[:, 1], uv[:, 0]] == depth
    if _config['use_reflectance']:
        refl_img = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float)
        refl_img[uv[indexes, 1], uv[indexes, 0]] = refl[0, indexes]

    depth_img_no_occlusion /= _config['max_depth']
    depth_img_no_occlusion = depth_img_no_occlusion.unsqueeze(0)

    if _config['use_reflectance']:
        depth_img_no_occlusion = torch.cat((depth_img_no_occlusion, refl_img.unsqueeze(0)))

    uv = uv[indexes]
    return depth_img_no_occlusion, uv, indexes, depth


def downsample_and_pad(_config, rgb, depth_img_no_occlusion, img_shape, real_shape, flow_img, flow_mask):
    shape_pad = [0, 0, 0, 0]

    if _config['dataset'] in ['argoverse', 'pandaset'] or _config['downsample']:
        rgb = nn.functional.interpolate(rgb.unsqueeze(0), scale_factor=0.5)[0]
        depth_img_no_occlusion = downsample_depth(depth_img_no_occlusion.permute(1, 2, 0).contiguous(), 2)
        depth_img_no_occlusion = depth_img_no_occlusion.permute(2, 0, 1)

        shape_pad[3] = (img_shape[0] - real_shape[0] // 2)  # // 2
        shape_pad[1] = (img_shape[1] - real_shape[1] // 2)  # // 2 + 1

        rgb = F.pad(rgb, shape_pad)
        depth_img_no_occlusion = F.pad(depth_img_no_occlusion, shape_pad)

        shape_pad[3] = (img_shape[0] * 2 - real_shape[0])  # // 2
        shape_pad[1] = (img_shape[1] * 2 - real_shape[1])  # // 2 + 1
        flow_img = F.pad(flow_img.permute(2, 0, 1), shape_pad).permute(1, 2, 0).contiguous()
        flow_mask = F.pad(flow_mask, shape_pad)

    else:
        shape_pad[3] = (img_shape[0] - real_shape[0])  # // 2
        shape_pad[1] = (img_shape[1] - real_shape[1])  # // 2 + 1

        rgb = F.pad(rgb, shape_pad)
        depth_img_no_occlusion = F.pad(depth_img_no_occlusion, shape_pad)
        flow_img = F.pad(flow_img.permute(2, 0, 1), shape_pad).permute(1, 2, 0).contiguous()
        flow_mask = F.pad(flow_mask, shape_pad)

    # Convert depth into fourier frequencies, similar to the positional encoding used in NERF
    if _config['fourier_levels'] >= 0:
        depth_img_no_occlusion = depth_img_no_occlusion.squeeze()
        mask = (depth_img_no_occlusion > 0).clone()
        fourier_feats = []
        for L in range(_config['fourier_levels']):
            fourier_feat = depth_img_no_occlusion * np.pi * 2 ** L
            fourier_feats.append(fourier_feat.sin())
            fourier_feats.append(fourier_feat.cos())
        depth_img_no_occlusion = torch.stack(fourier_feats + [depth_img_no_occlusion])
        depth_img_no_occlusion = depth_img_no_occlusion * mask.unsqueeze(0)

    return rgb, depth_img_no_occlusion, flow_img, flow_mask


# noinspection PyUnreachableCode
def evaluate_calibration(_config, seed):
    global EPOCH

    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)
    cv2.setRNGSeed(seed)
    if _config['deterministic']:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

    checkpoint = torch.load(_config['weights'][0], map_location='cpu')

    if _config['viz']:
        f, axarr = plt.subplots(2, 1)
        axarr[0].set_title('Initial Calibration')
        axarr[1].set_title('CMRNext Estimated Calibration')
        plt.show(block=False)
        plt.pause(1)

    _config['network'] = 'RAFT'
    _config['use_reflectance'] = checkpoint['config']['use_reflectance']
    _config['initial_pool'] = True
    _config['upsample_method'] = checkpoint['config']['upsample_method']
    _config['occlusion_kernel'] = 9
    _config['occlusion_threshold'] = 3.9999
    _config['scaled_gt'] = False
    _config['uncertainty'] = False
    _config['der_type'] = "NLL"
    _config['unc_freeze'] = False
    _config['normalize_images'] = checkpoint['config']['normalize_images']
    _config['max_depth'] = checkpoint['config']['max_depth']
    if 'uncertainty' in checkpoint['config']:
        _config['uncertainty'] = checkpoint['config']['uncertainty']
    _config['fourier_levels'] = -1
    if 'fourier_levels' in checkpoint['config']:
        _config['fourier_levels'] = checkpoint['config']['fourier_levels']
    _config['num_scales'] = 1
    if 'num_scales' in checkpoint['config']:
        _config['num_scales'] = checkpoint['config']['num_scales']
    if 'der_type' in checkpoint['config']:
        _config['der_type'] = checkpoint['config']['der_type']
    if 'unc_freeze' in checkpoint['config']:
        _config['unc_freeze'] = checkpoint['config']['unc_freeze']
    _config['context_encoder'] = 'rgb'
    if 'context_encoder' in checkpoint['config']:
        _config['context_encoder'] = checkpoint['config']['context_encoder']

    mean_torch = torch.tensor([0.485, 0.456, 0.406]).to(device)
    std_torch = torch.tensor([0.229, 0.224, 0.225]).to(device)

    # Setup Dataset
    val_directories = []
    base_dir = _config['data_folder']
    if _config['dataset'] == 'argoverse':
        img_shape = (640, 1920 // 2)  # Multiple of 64, inference at scale 0.5
        if _config['cam'] is None:
            _config['cam'] = 'ring_front_center'
        else:
            assert _config['cam'] in ['ring_front_center'], \
                f"Camera {_config['cam']} not supported for the {_config['dataset']} dataset"
    elif 'kitti' in _config['dataset']:
        img_shape = (384, 1280)  # Multiple of 64
        for subdir in ['00']:
            val_directories.append(os.path.join(base_dir, subdir))
        if _config['cam'] is None:
            _config['cam'] = '2'
        else:
            assert str(_config['cam']) in ['2', '3'], \
                f"Camera {_config['cam']} not supported for the {_config['dataset']} dataset"
    elif _config['dataset'] == 'pandaset':
        img_shape = (576, 1920 // 2)  # Multiple of 64 inference at scale 0.5
        if _config['cam'] is None:
            _config['cam'] = 'front_camera'
        else:
            assert _config['cam'] in ['front_camera'], \
                f"Camera {_config['cam']} not supported for the {_config['dataset']} dataset"
    elif _config['dataset'] == 'custom':
        val_directories.append(base_dir)
        first_camera_path = os.listdir(os.path.join(_config['data_folder'], 'camera'))[0]
        first_camera_frame = np.asarray(Image.open(os.path.join(_config['data_folder'], 'camera', first_camera_path)))
        img_shape = [first_camera_frame.shape[0], first_camera_frame.shape[1]]
        if _config['downsample']:
            img_shape = [img_shape[0] // 2, img_shape[1] // 2]
        if img_shape[0] % 64 > 0:
            img_shape[0] = 64 * ((img_shape[0] // 64) + 1)
        if img_shape[1] % 64 > 0:
            img_shape[1] = 64 * ((img_shape[1] // 64) + 1)
    elif _config['dataset'] == 'hercules':
        dataset_val = DatasetHerculesRadarExtrinsicCalib(
            base_dir,
            train=False,
            max_r=_config['max_r'],
            max_t=_config['max_t'],
            use_reflectance=_config['use_reflectance'],
            normalize_images=_config['normalize_images'],
            val_scene=_config['val_scene'],
        )
        img_shape = infer_hercules_img_shape(dataset_val)
        if _config['downsample']:
            img_shape = [img_shape[0] // 2, img_shape[1] // 2]
            img_shape[0] = 64 * ((img_shape[0] + 63) // 64)
            img_shape[1] = 64 * ((img_shape[1] + 63) // 64)
    else:
        raise RuntimeError("Dataset unknown")

    if _config['dataset'] == 'kitti':
        dataset_val = DatasetGeneralExtrinsicCalib(val_directories, train=False, max_r=_config['max_r'],
                                                   max_t=_config['max_t'], use_reflectance=_config['use_reflectance'],
                                                   normalize_images=_config['normalize_images'],
                                                   dataset=_config['dataset'], cam=_config['cam'])
    elif _config['dataset'] == 'argoverse':
        for subdir in ['train4']:
            val_directories.append(os.path.join(_config['data_folder'], subdir))
        dataset_val = DatasetGeneralExtrinsicCalib(val_directories, train=False, max_r=_config['max_r'],
                                                   max_t=_config['max_t'], use_reflectance=_config['use_reflectance'],
                                                   normalize_images=_config['normalize_images'],
                                                   dataset=_config['dataset'], cam=_config['cam'])
    elif _config['dataset'] == 'pandaset':
        for subdir in ['011', '122', '124', '030', '109', '043', '084', '115', '090']:
            val_directories.append(os.path.join(_config['data_folder'], subdir))
        dataset_val = DatasetPandasetExtrinsicCalib(val_directories, train=False, max_r=_config['max_r'],
                                                    max_t=_config['max_t'],
                                                    use_reflectance=_config['use_reflectance'],
                                                    normalize_images=_config['normalize_images'],
                                                    sensor_id=0, camera=_config['cam'])
    elif _config['dataset'] == 'custom':
        dataset_val = DatasetGeneralExtrinsicCalib(val_directories, train=False, max_r=_config['max_r'],
                                                   max_t=_config['max_t'], use_reflectance=_config['use_reflectance'],
                                                   normalize_images=_config['normalize_images'],
                                                   dataset=_config['dataset'], cam=_config['cam'])
    elif _config['dataset'] == 'hercules':
        pass

    def init_fn(x):
        return _init_fn(x, seed)

    # Training and test set creation
    num_worker = _config['num_worker']
    batch_size = 1  # This code is designed for a batch size of 1, don't change this!

    TestImgLoader = torch.utils.data.DataLoader(dataset=dataset_val,
                                                shuffle=False,
                                                batch_size=batch_size,
                                                num_workers=num_worker,
                                                worker_init_fn=init_fn,
                                                collate_fn=merge_inputs,
                                                drop_last=False,
                                                pin_memory=False)

    print(len(TestImgLoader))

    models = []
    print(f"Loading weights from {_config['weights']}")
    for i in range(len(_config['weights'])):
        checkpoint = torch.load(_config['weights'][i], map_location='cpu')
        model = get_model(_config, img_shape)
        saved_state_dict = checkpoint['state_dict']
        clean_state_dict = saved_state_dict
        model.load_state_dict(clean_state_dict, strict=False)
        model = model.to(device)
        model.eval()
        models.append(model)

    errors_r = []
    errors_t = []
    list_quats = []
    list_transl = []
    epe = []
    ransac_time = []
    inference_time = []
    final_calib_RTs = []
    for i in range(len(_config['weights']) + 1):
        errors_r.append([])
        errors_t.append([])
        list_quats.append([])
        list_transl.append([])
        epe.append([])
        final_calib_RTs.append([])
    tbar = tqdm(TestImgLoader)
    for batch_idx, sample in enumerate(tbar):
        lidar_input = []
        rgb_input = []

        sample['tr_error'] = sample['tr_error'].cuda()
        sample['rot_error'] = sample['rot_error'].cuda()

        for idx in range(len(sample['rgb'])):
            errors_r[0].append(quaternion_distance(sample['rot_error'][idx],
                                                   torch.tensor([1., 0., 0., 0.], device=sample['rot_error'].device)))

            errors_t[0].append(sample['tr_error'][idx].norm().item())
            list_quats[0].append(sample['rot_error'][idx].cpu().numpy())
            list_transl[0].append(sample['tr_error'][idx].cpu().numpy())

            # ProjectPointCloud in RT-pose

            real_shape = [sample['rgb'][idx].shape[0], sample['rgb'][idx].shape[1], sample['rgb'][idx].shape[2]]

            sample['point_cloud'][idx] = sample['point_cloud'][idx].cuda()
            if _config['max_depth'] < 100.:
                sample['point_cloud'][idx] = sample['point_cloud'][idx][:,
                                             sample['point_cloud'][idx][0, :] < _config['max_depth']]

            pc_rotated = sample['point_cloud'][idx].clone()
            reflectance = None
            if _config['use_reflectance']:
                reflectance = sample['reflectance'][idx].cuda()

            R = quat2mat(sample['rot_error'][idx])
            T = tvector2mat(sample['tr_error'][idx])
            RT1_inv = torch.mm(T, R)
            extrinsic_error = [RT1_inv.clone().inverse()]
            extrinsic_prediction = [torch.eye(4)]

            pc_rotated = rotate_forward(pc_rotated, extrinsic_error[0])

            # Project point cloud into virtual image plane placed at random 'initial_calib'
            cam_params = sample['calib'][idx].cuda()
            depth_img_no_occlusion, uv, indexes, depth = prepare_input(cam_params, pc_rotated, real_shape,
                                                                       reflectance, _config)
            cam_model = CameraModel()
            cam_model.focal_length = cam_params[:2]
            cam_model.principal_point = cam_params[2:]

            flow, points_3D, new_indexes = get_flow_zforward(uv.float(), depth[indexes], RT1_inv, cam_model,
                                                             [real_shape[0], real_shape[1], 3],
                                                             scale_flow=False, reverse=False,
                                                             get_valid_indexes=True)

            uv = uv[new_indexes].clone()
            flow = flow[new_indexes].clone()

            rgb = sample['rgb'][idx].cuda()

            # Normalize image
            rgb = rgb / 255.
            if _config['normalize_images']:
                rgb = (rgb - mean_torch) / std_torch
            rgb = rgb.permute(2, 0, 1)
            sample['rgb'][idx] = rgb

            flow_img = torch.zeros((real_shape[0], real_shape[1], 2), device='cuda', dtype=torch.float)
            flow_img[uv[:, 1], uv[:, 0]] = flow
            # flow_mask containts 1 in pixels that have a point projected
            flow_mask = torch.zeros((real_shape[0], real_shape[1]), device='cuda', dtype=torch.int)
            flow_mask[uv[:, 1], uv[:, 0]] = 1

            if _config['viz']:
                viz_initial = overlay_imgs(rgb, depth_img_no_occlusion[-1].unsqueeze(0).unsqueeze(0), max_depth=0.5,
                                           close_thr=1000)
                # plt.imshow(viz_initial)

            points_3D = points_3D[new_indexes].clone()
            rgb, depth_img_no_occlusion, flow_img, flow_mask = downsample_and_pad(_config, rgb, depth_img_no_occlusion,
                                                                                  img_shape, real_shape, flow_img,
                                                                                  flow_mask)

            rgb_input.append(rgb)
            lidar_input.append(depth_img_no_occlusion)

        lidar_input = torch.stack(lidar_input)
        rgb_input = torch.stack(rgb_input)

        for iteration in range(len(_config['weights'])):
            torch.cuda.synchronize()
            time1 = time.time()
            # Predict 'flow': dense lidar depth map to rgb pixel displacements
            with torch.no_grad():
                predicted_flow = models[iteration](rgb_input, lidar_input)
                torch.cuda.synchronize()
                time2 = time.time()
                predicted_flow, predicted_uncertainty = predicted_flow
                inference_time.append(time2 - time1)
                # Upsample if necessary
                if _config['dataset'] in ['argoverse', 'pandaset'] or _config['downsample']:
                    predicted_flow = list(predicted_flow)
                    predicted_flow[-1] *= 2
                    predicted_flow[-1] = F.interpolate(predicted_flow[-1], scale_factor=2, mode='bilinear')
                    if _config['uncertainty']:
                        predicted_uncertainty[-1] = F.interpolate(predicted_uncertainty[-1], scale_factor=2,
                                                                  mode='bilinear')

            up_flow = predicted_flow[-1]

            # EPE
            gt = flow_img.clone().permute(2, 0, 1)
            gt = torch.cat((gt, flow_mask.unsqueeze(0).float()))
            gt = gt.unsqueeze(0)
            epe[iteration].append(EndPointError(up_flow, gt).item())

            up_flow = up_flow[0].permute(1, 2, 0)

            new_uv = uv.float() + up_flow[uv[:, 1], uv[:, 0]]

            valid_indexes = flow_mask[uv[:, 1], uv[:, 0]] == 1

            if flow_mask.sum() < 10:
                break

            if _config['uncertainty']:
                sum_uncertainty = predicted_uncertainty[-1][0, 0] + predicted_uncertainty[-1][0, 1]
                mean_uncertainty = sum_uncertainty * flow_mask
                try:
                    mean_uncertainty = np.quantile(mean_uncertainty[flow_mask != 0].detach().cpu().numpy(),
                                                   _config['quantile'])
                except:
                    pass
                valid_indexes = valid_indexes & (sum_uncertainty[uv[:, 1], uv[:, 0]] < mean_uncertainty)

            # Check only pixels that are within the image border
            valid_indexes = valid_indexes & (new_uv[:, 0] < flow_mask.shape[1])
            valid_indexes = valid_indexes & (new_uv[:, 1] < flow_mask.shape[0])
            valid_indexes = valid_indexes & (new_uv[:, 0] >= 0)
            valid_indexes = valid_indexes & (new_uv[:, 1] >= 0)
            new_uv = new_uv[valid_indexes]

            valid_indexes2 = torch.ones(new_uv.shape[0], dtype=torch.bool).cuda()

            new_uv = new_uv[valid_indexes2]

            points_2d = new_uv.cpu().numpy()
            obj_coord = points_3D[valid_indexes][valid_indexes2][:, :3].cpu().numpy()
            obj_coord_zforward = np.zeros(obj_coord.shape)
            obj_coord_zforward[:, 0] = obj_coord[:, 0]
            obj_coord_zforward[:, 1] = obj_coord[:, 1]
            obj_coord_zforward[:, 2] = obj_coord[:, 2]
            cam_mat = cam_model.get_matrix()

            torch.cuda.synchronize()

            time1 = time.time()
            if obj_coord_zforward.shape[0] < 10:
                for left_iter in range(iteration + 1):
                    errors_t[left_iter].pop(-1)
                    errors_r[left_iter].pop(-1)
                    list_transl[left_iter].pop(-1)
                    list_quats[left_iter].pop(-1)
                break

            if _config['quantile'] < 1.0 and not _config['uncertainty']:
                num_corr = points_2d.shape[0]
                corr_to_keep = list(range(num_corr))
                random.shuffle(corr_to_keep)
                num_corr_to_keep = int(num_corr * _config['quantile'])
                if num_corr_to_keep > 10:
                    points_2d = points_2d[corr_to_keep[:num_corr_to_keep]]
                    obj_coord_zforward = obj_coord_zforward[corr_to_keep[:num_corr_to_keep]]

            # Predict relative transformation based on CMRNext correspondences
            # for iterative refinement
            cuda_pnp = cv2.pythoncuda.cudaPnP(obj_coord_zforward.astype(np.float32).copy(),
                                              points_2d.astype(np.float32).copy(), obj_coord_zforward.shape[0],
                                              200, 2., cam_mat.astype(np.float32)
                                              )

            transl = cuda_pnp[0, [0, 1, 2]]
            rot_mat = cuda_pnp[:, 3:6].T
            rot_mat, _ = cv2.Rodrigues(rot_mat)

            torch.cuda.synchronize()
            time2 = time.time()
            # print("Ransac: ", time2-time1)
            ransac_time.append(time2 - time1)

            transl = torch.tensor(transl).float().squeeze().cuda()

            rot_mat = torch.tensor(rot_mat)

            pred_quaternion = quaternion_from_matrix(rot_mat)

            R_predicted = quat2mat(pred_quaternion).cuda()
            T_predicted = tvector2mat(transl)
            RT_predicted = torch.mm(T_predicted, R_predicted)
            composed = torch.mm(extrinsic_error[iteration], RT_predicted.inverse())
            extrinsic_error.append(composed)
            extrinsic_prediction.append(torch.mm(extrinsic_prediction[iteration], RT_predicted.inverse().cpu()))

            T_composed = composed[:3, 3]
            R_composed = quaternion_from_matrix(composed)
            errors_r[iteration + 1].append(quaternion_distance(R_composed,
                                                               torch.tensor([1., 0., 0., 0.],
                                                                            device=R_composed.device)))
            errors_t[iteration + 1].append(T_composed.norm().item())

            list_transl[iteration + 1].append(T_composed.cpu().numpy())
            list_quats[iteration + 1].append(R_composed.cpu().numpy())

            # Compute final cam lidar predicted matrix
            points_3D_orig = torch.mm(extrinsic_prediction[-2].to(points_3D.device), points_3D[valid_indexes].T).T
            points_3D_orig = torch.mm(extrinsic_error[0], points_3D_orig.T).T
            # points_3D_orig = rotate_back(points_3D_orig, sample['cam2vel'][0].to(points_3D_orig.device))
            points_3D_orig = rotate_forward(points_3D_orig, sample['cam2vel'][0].to(points_3D_orig.device))
            final_correspondences = new_uv, points_3D_orig
            cuda_pnp_final = cv2.pythoncuda.cudaPnP(
                final_correspondences[1][:, :3].cpu().numpy().astype(np.float32).copy(),
                final_correspondences[0].cpu().numpy().astype(np.float32).copy(),
                points_3D_orig.shape[0], 200, 2., cam_mat.astype(np.float32))
            transl_final = cuda_pnp_final[0, :3]
            rot_mat_final = cuda_pnp_final[:, 3:6].T
            rot_mat_final, _ = cv2.Rodrigues(rot_mat_final)
            rot_mat_final = torch.tensor(rot_mat_final)
            T_predicted_final = tvector2mat(torch.tensor(transl_final))
            R_predicted_final = torch.eye(4)
            R_predicted_final[:3, :3] = rot_mat_final.clone().detach()
            # Final prediction by CMRNet (for each iteration)
            RT_predicted_final = torch.mm(T_predicted_final, R_predicted_final)
            # final_calib_RTs[iteration + 1].append(RT_predicted_final.inverse())
            final_calib_RTs[iteration + 1].append(RT_predicted_final)

            if T_composed.norm().item() > 4.:
                # Prediction has failed for this frame
                for left_iteration in range(iteration + 2, len(_config['weights']) + 1):
                    errors_t[left_iteration].append(T_composed.norm().item())
                    errors_r[left_iteration].append(errors_r[iteration + 1][-1])
                break

            # Rotate point cloud based on predicted pose, and generate new
            # inputs for the next iteration
            rotated_point_cloud = rotate_forward(sample['point_cloud'][idx], extrinsic_error[-1])

            depth_img_no_occlusion, uv, indexes, depth = prepare_input(cam_params, rotated_point_cloud, real_shape,
                                                                       reflectance, _config)

            flow, points_3D, new_indexes = get_flow_zforward(uv.float(), depth[indexes], extrinsic_error[-1].inverse(),
                                                             cam_model, [real_shape[0], real_shape[1], 3],
                                                             scale_flow=False, reverse=False,
                                                             get_valid_indexes=True)

            uv = uv[new_indexes].clone()
            flow = flow[new_indexes].clone()

            rgb = sample['rgb'][idx].cuda()
            flow_img = torch.zeros((real_shape[0], real_shape[1], 2), device='cuda', dtype=torch.float)
            flow_img[uv[:, 1], uv[:, 0]] = flow
            flow_mask = torch.zeros((real_shape[0], real_shape[1]), device='cuda', dtype=torch.int)
            flow_mask[uv[:, 1], uv[:, 0]] = 1

            points_3D = points_3D[new_indexes].clone()
            rgb, depth_img_no_occlusion, flow_img, flow_mask = downsample_and_pad(_config, rgb, depth_img_no_occlusion,
                                                                                  img_shape, real_shape,
                                                                                  flow_img, flow_mask)

            rgb_input = rgb.unsqueeze(0)
            lidar_input = depth_img_no_occlusion.unsqueeze(0)

            if _config['viz'] and iteration == len(_config['weights']) - 1:
                gt_uv, gt_depth, _, _ = cam_model.project_pytorch(rotated_point_cloud, real_shape, reflectance)
                gt_uv = gt_uv.t().int().contiguous()

                new_depth_img = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float)
                new_depth_img += 1000.
                new_depth_img = visibility.depth_image(gt_uv.int().contiguous(), gt_depth, new_depth_img,
                                                       gt_uv.shape[0],
                                                       real_shape[1], real_shape[0])
                new_depth_img[new_depth_img == 1000.] = 0.

                new_depth_img_no_occlusion = torch.zeros_like(new_depth_img, device='cuda')
                new_depth_img_no_occlusion = visibility.visibility2(new_depth_img, cam_params,
                                                                    new_depth_img_no_occlusion,
                                                                    new_depth_img.shape[1], new_depth_img.shape[0],
                                                                    _config['occlusion_threshold'],
                                                                    _config['occlusion_kernel'])

                lidar_flow = new_depth_img_no_occlusion.unsqueeze(0).unsqueeze(0)
                viz_final = overlay_imgs(sample['rgb'][idx].cuda(), lidar_flow, max_depth=_config['max_depth'] / 2,
                                         close_thr=1000)

                axarr[0].imshow(viz_initial)
                axarr[1].imshow(viz_final)
                plt.draw()
                plt.pause(5)

            try:
                if _config['dataset'] != 'custom':
                    tbar.set_postfix(t_mean=torch.tensor(errors_t[-1]).mean().item(),
                                     t_median=torch.tensor(errors_t[-1]).median().item(),
                                     r_mean=torch.tensor(errors_r[-1]).mean().item(),
                                     r_median=torch.tensor(errors_r[-1]).median().item(),
                                     epe_mean=torch.tensor(epe[0]).mean().item())
            except:
                pass

    print(f'Network Time: ', torch.tensor(inference_time).mean())
    print(f'PnP+RANSAC Time: ', torch.tensor(ransac_time).mean())
    for iteration in range(len(_config['weights']) + 1):
        errors_t[iteration] = torch.tensor(errors_t[iteration])
        errors_r[iteration] = torch.tensor(errors_r[iteration])

    if _config['dataset'] == 'custom' and len(final_calib_RTs[len(_config['weights'])]) == 1:
        print("Predicted extrinsic calibration:")
        print(final_calib_RTs[len(_config['weights'])][0])
    elif _config['dataset'] == 'custom':
        iteration = len(_config['weights'])
        final_quats = np.stack([quaternion_from_matrix(t) for t in final_calib_RTs[iteration]])
        avg_quaternion = torch.from_numpy(average_quaternions(final_quats))
        median_quaternion = quaternion_median(torch.tensor(np.stack(final_quats)))
        mode_quaternion = quaternion_mode(final_quats, 4)

        avg_translation = torch.stack(final_calib_RTs[iteration])[:, :3, 3].mean(0)
        median_translation = torch.stack(final_calib_RTs[iteration])[:, :3, 3].median(0)[0]
        mode_translation = quaternion_mode(torch.stack(final_calib_RTs[iteration])[:, :3, 3], 2)

        R = quat2mat(avg_quaternion)
        T = tvector2mat(avg_translation)
        avg_extrinsic_calib = torch.mm(T, R)
        torch.set_printoptions(5, sci_mode=False)
        print("Predicted extrinsic calibration using average aggregation:")
        print(avg_extrinsic_calib)

        R = quat2mat(median_quaternion)
        T = tvector2mat(median_translation)
        median_extrinsic_calib = torch.mm(T, R)
        torch.set_printoptions(5, sci_mode=False)
        print("Predicted extrinsic calibration using median aggregation:")
        print(median_extrinsic_calib)

        R = quat2mat(mode_quaternion)
        T = tvector2mat(mode_translation)
        mode_extrinsic_calib = torch.mm(T, R)
        print("Predicted extrinsic calibration using mode aggregation:")
        print(mode_extrinsic_calib)

    else:
        console = Console()
        table = Table(show_header=True, header_style="bold magenta", box=box.MINIMAL_HEAVY_HEAD, title_style="bold red")
        table.title = f"CMRNext Results on {_config['dataset']}, camera {_config['cam']}"
        table.add_column("Iteration")
        table.add_column("Median Translation error (cm)", justify="center", max_width=20)
        table.add_column("Median Rotation error (˚)", justify="center", max_width=20)
        table.add_row(
            f"Initial Pose",
            f"{errors_t[0].median().item() * 100:.2f}",
            f"{errors_r[0].median().item():.2f}"
        )
        for iteration in range(1, len(_config['weights']) + 1):
            table.add_row(
                f"Iteration {iteration}",
                f"{errors_t[iteration].median().item() * 100:.2f}",
                f"{errors_r[iteration].median().item():.2f}"
            )
        print("")
        print("")
        console.print(table)

        table = Table(show_header=True, header_style="bold magenta", box=box.MINIMAL_HEAVY_HEAD, title_style="bold red")
        table.title = f"Temporal Aggregation Results on {_config['dataset']}"
        table.add_column("Aggregation Measure", max_width=13)
        table.add_column("Translation Error (cm)", justify="center")
        table.add_column("Rotation Error (˚)", justify="center")

        iteration = len(_config['weights'])
        final_quats = np.stack([quaternion_from_matrix(t) for t in final_calib_RTs[iteration]])
        r_error_avg = quaternion_distance(
            torch.from_numpy(average_quaternions(final_quats)),
            quaternion_from_matrix(sample['cam2vel'][0])
        )
        # r_error_median = quaternion_distance(
        #     quaternion_median(np.stack(final_quats)),
        #     quaternion_from_matrix(sample['cam2vel'][0])
        # )
        r_error_mode = quaternion_distance(
            quaternion_mode(final_quats, 4),
            quaternion_from_matrix(sample['cam2vel'][0])
        )
        if r_error_mode > quaternion_distance(
                quaternion_mode(final_quats, 3),
                quaternion_from_matrix(sample['cam2vel'][0])
        ):
            r_error_mode = quaternion_distance(
                quaternion_mode(final_quats, 3),
                quaternion_from_matrix(sample['cam2vel'][0])
            )

        t_error_avg = (torch.stack(final_calib_RTs[iteration])[:, :3, 3].mean(0)
                       - sample['cam2vel'][0][:3, 3]).norm() * 100.
        t_error_median = (torch.stack(final_calib_RTs[iteration])[:, :3, 3].median(0)[0]
                          - sample['cam2vel'][0][:3, 3]).norm() * 100.
        t_error_mode = (quaternion_mode(torch.stack(final_calib_RTs[iteration])[:, :3, 3], 2)
                        - sample['cam2vel'][0][:3, 3]).norm() * 100.
        if t_error_mode > (quaternion_mode(torch.stack(final_calib_RTs[iteration])[:, :3, 3], 1)
                           - sample['cam2vel'][0][:3, 3]).norm() * 100.:
            t_error_mode = (quaternion_mode(torch.stack(final_calib_RTs[iteration])[:, :3, 3], 1)
                            - sample['cam2vel'][0][:3, 3]).norm() * 100.
        table.add_row(
            "Mean",
            f"[bold green]{t_error_avg.item():.2f}[/bold green]" if t_error_avg.item() <= t_error_median.item() and
                                                                    t_error_avg.item() <= t_error_mode.item() else
            f"{t_error_avg.item():.2f}",
            f"[bold green]{r_error_avg:.2f}[/bold green]" if r_error_avg <= r_error_mode else
            f"{r_error_avg:.2f}",
        )
        table.add_row(
            "Median",
            f"[bold green]{t_error_median.item():.2f}[/bold green]" if t_error_median.item() <= t_error_avg.item() and
                                                                       t_error_median.item() <= t_error_mode.item() else
            f"{t_error_median.item():.2f}",
            f"---"
        )
        table.add_row(
            "Mode",
            f"[bold green]{t_error_mode.item():.2f}[/bold green]" if t_error_mode.item() <= t_error_avg.item() and
                                                                     t_error_mode.item() <= t_error_median.item() else
            f"{t_error_mode.item():.2f}",
            f"[bold green]{r_error_mode:.2f}[/bold green]" if r_error_mode <= r_error_avg else
            f"{r_error_mode:.2f}"
        )
        print("")
        print("")
        console.print(table)

    if _config['save_file'] is not None:
        torch.save(errors_t, f'./{_config["save_file"]}_errors_t.torch')
        torch.save(errors_r, f'./{_config["save_file"]}_errors_r.torch')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='kitti')
    parser.add_argument('--data_folder', type=str, default='/data/KITTI/sequences/')
    parser.add_argument('--cam', type=str, nargs='?', default=None)
    parser.add_argument('--max_t', type=float, default=1.5)
    parser.add_argument('--max_r', type=float, default=20.)
    parser.add_argument('--num_worker', type=int, default=2)
    parser.add_argument('--weights', type=str, nargs='+', default=None)
    parser.add_argument('--img_shape', type=int, nargs=1, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--deterministic', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--save_file', type=str, nargs='?', default=None)
    parser.add_argument('--quantile', type=float, default=1.0)
    parser.add_argument('--downsample', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--viz', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--val_scene', type=str, nargs='*', default=None)

    args = parser.parse_args()
    _config = vars(args)
    evaluate_calibration(_config, _config['seed'])


if __name__ == '__main__':
    main()
