import json
import logging
import os
import sys
from math import radians

import cv2
import mathutils
import numpy as np
import pandas as pd
import pykitti
import pyntcloud
import torch
import yaml
from PIL import Image
from argoverse.data_loading.synchronization_database import SynchronizationDB
from argoverse.utils.calibration import get_calibration_config
from argoverse.utils.json_utils import read_json_file
from pandaset.geometry import _heading_position_to_mat
from torch.utils.data import Dataset
from torchvision import transforms

from utils import invert_pose, rotate_forward, to_rotation_matrix

logging.getLogger('argoverse').setLevel(logging.ERROR)


def is_image(img):
    extensions = ['.jpg', '.png', '.tiff', '.jpeg', '.bmp']
    return os.path.splitext(img)[1] in extensions


def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data


def _load_sync_pairs(pair_file, image_dir, sensor_dir):
    def _index_files(directory):
        file_map = {}
        for name in sorted(os.listdir(directory)):
            path = os.path.join(directory, name)
            if not os.path.isfile(path):
                continue
            stem, _ = os.path.splitext(name)
            file_map[stem] = path
            file_map[name] = path
            if stem.isdigit():
                file_map[str(int(stem))] = path
        return file_map

    def _match_path(file_map, token):
        candidates = [token]
        stem = os.path.splitext(token)[0]
        if stem not in candidates:
            candidates.append(stem)
        if token.isdigit():
            candidates.append(str(int(token)))
        if stem.isdigit():
            candidates.append(str(int(stem)))
        for candidate in candidates:
            if candidate in file_map:
                return file_map[candidate]
        return None

    image_map = _index_files(image_dir)
    sensor_map = _index_files(sensor_dir)
    pairs = []

    with open(pair_file, 'r') as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            tokens = line.replace(',', ' ').split()
            if len(tokens) < 2:
                continue

            # Skip header rows like "image_Cam0 radar_Continental".
            if line_idx == 0 and (not tokens[0].isdigit() or not tokens[1].isdigit()):
                continue

            if not tokens[0].isdigit() or not tokens[1].isdigit():
                continue

            image_path = _match_path(image_map, tokens[0])
            sensor_path = _match_path(sensor_map, tokens[1])
            if image_path is None or sensor_path is None:
                continue

            pairs.append({
                'stamp': tokens[0],
                'image_path': image_path,
                'sensor_path': sensor_path,
                'image_name': os.path.basename(image_path),
            })

    return pairs


def _read_pcd(file_path):
    type_map = {
        ('F', 4): np.float32,
        ('F', 8): np.float64,
        ('I', 1): np.int8,
        ('I', 2): np.int16,
        ('I', 4): np.int32,
        ('I', 8): np.int64,
        ('U', 1): np.uint8,
        ('U', 2): np.uint16,
        ('U', 4): np.uint32,
        ('U', 8): np.uint64,
    }

    header = {}
    header_lines = []
    data_start = 0
    with open(file_path, 'rb') as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Invalid PCD header: {file_path}")
            data_start += len(line)
            line_str = line.decode('ascii', errors='ignore').strip()
            header_lines.append(line_str)
            if not line_str or line_str.startswith('#'):
                continue
            parts = line_str.split()
            key = parts[0].upper()
            values = parts[1:]
            header[key] = values
            if key == 'DATA':
                break

        fields = header.get('FIELDS')
        sizes = [int(v) for v in header.get('SIZE', [])]
        types = header.get('TYPE', [])
        counts = [int(v) for v in header.get('COUNT', ['1'] * len(fields))]
        width = int(header.get('WIDTH', ['0'])[0])
        height = int(header.get('HEIGHT', ['1'])[0])
        points = int(header.get('POINTS', [str(width * height)])[0])
        data_kind = header['DATA'][0].lower()

        if not fields or not sizes or not types:
            raise ValueError(f"Incomplete PCD header: {file_path}")
        if not (len(fields) == len(sizes) == len(types) == len(counts)):
            raise ValueError(f"Mismatched PCD header lengths: {file_path}")

        dtype_fields = []
        for name, size, typ, count in zip(fields, sizes, types, counts):
            key = (typ.upper(), size)
            if key not in type_map:
                raise ValueError(f"Unsupported PCD field type {key} in {file_path}")
            base_dtype = np.dtype(type_map[key])
            if count == 1:
                dtype_fields.append((name, base_dtype))
            else:
                dtype_fields.append((name, base_dtype, (count,)))
        dtype = np.dtype(dtype_fields)

        if data_kind == 'binary':
            raw = f.read(points * dtype.itemsize)
            if len(raw) < points * dtype.itemsize:
                raise ValueError(f"Unexpected EOF in binary PCD: {file_path}")
            data = np.frombuffer(raw, dtype=dtype, count=points)
        elif data_kind == 'ascii':
            f.seek(data_start)
            data = np.loadtxt(f, dtype=dtype, comments='#')
            if data.shape == ():
                data = np.array([data], dtype=dtype)
        else:
            raise ValueError(f"Unsupported PCD DATA mode '{data_kind}' in {file_path}")

    xyz = np.stack([data['x'], data['y'], data['z']], axis=1).astype(np.float32, copy=False)
    if 'intensity' in data.dtype.names:
        intensity = np.asarray(data['intensity'], dtype=np.float32).reshape(-1, 1)
        return np.concatenate((xyz, intensity), axis=1)
    return xyz


def _retry_with_different_sample(dataset, idx, exc, sample_path=None, max_attempts=10):
    dataset_len = len(dataset)
    if dataset_len <= 1:
        raise RuntimeError(f"Failed to load sample {sample_path or idx}: {exc}") from exc

    for _ in range(max_attempts):
        new_idx = np.random.randint(0, dataset_len)
        if new_idx != idx:
            return dataset.__getitem__(new_idx)

    raise RuntimeError(f"Failed to resample after unreadable sample: {sample_path or idx}") from exc


class ReadKITTI:
    def __call__(self, file):
        return np.fromfile(file, dtype=np.float32).reshape((-1, 4))


class ReadPCD:
    def __call__(self, file):
        return _read_pcd(file)


def _list_point_cloud_files(directory, extensions=('pcd', 'bin')):
    point_cloud_files = []
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        extension = os.path.splitext(name)[1].lstrip('.').lower()
        if extension in extensions:
            point_cloud_files.append(name)
    return point_cloud_files


# Generic point cloud reader from https://github.com/PRBonn/kiss-icp
def _get_point_cloud_reader(file_extension, first_scan_file):
        """Attempt to guess with try/catch blocks which is the best point cloud reader to use for
        the given dataset folder. Supported readers so far are:
            - np.fromfile
            - trimesh.load
            - PyntCloud
            - open3d[optional]
        """
        # This is easy, the old KITTI format
        if file_extension == "bin":
            print("[WARNING] Reading .bin files, the only format supported is the KITTI format")

            return ReadKITTI()

        if file_extension == "pcd":
            print("Using native reader for .pcd data")

            # Probe the first file once so failures surface immediately.
            ReadPCD()(first_scan_file)
            return ReadPCD()

        print('Trying to guess how to read your data')
        # first try open3d
        try:
            import open3d as o3d

            try_pcd = o3d.io.read_point_cloud(first_scan_file)
            if try_pcd.is_empty():
                # open3d binding does not raise an exception if file is unreadable or extension is not supported
                raise Exception("Generic Dataloader| Open3d PointCloud file is empty")

            class ReadOpen3d:
                def __call__(self, file):
                    pcd = o3d.io.read_point_cloud(file)
                    points = np.asarray(pcd.points)
                    return points

            return ReadOpen3d()
        except:
            pass

        try:
            import trimesh

            trimesh.load(first_scan_file)

            class ReadTriMesh:
                def __call__(self, file):
                    return np.asarray(trimesh.load(file).vertices)

            return ReadTriMesh()
        except:
            pass

        try:
            from pyntcloud import PyntCloud

            PyntCloud.from_file(first_scan_file)

            class ReadPynt:
                def __call__(self, file):
                    return PyntCloud.from_file(file).points[["x", "y", "z"]].to_numpy()

            return ReadPynt()
        except:
            print("[ERROR], File format not supported")
            sys.exit(1)


def get_scan_kitti(path, cam='2', kitti=None):
    scan = np.fromfile(path, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    split_path = path.split('/')
    base_folder = os.path.join('/', *split_path[:-4])
    if kitti is None:
        kitti = pykitti.odometry(base_folder, split_path[-3])
    if cam == '2' or cam == '02':
        cam_to_velo = torch.from_numpy(kitti.calib.T_cam2_velo).double()
        calib = kitti.calib.K_cam2
    elif cam == '3' or cam == '03':
        cam_to_velo = torch.from_numpy(kitti.calib.T_cam3_velo).double()
        calib = kitti.calib.K_cam3
    calib = torch.tensor([calib[0, 0], calib[1, 1], calib[0, 2], calib[1, 2]]).float()
    return scan, cam_to_velo.float(), calib


def get_scan_argo(path, camera):
    data = pyntcloud.PyntCloud.from_file(os.fspath(path))
    x = np.array(data.points.x)[:, np.newaxis]
    y = np.array(data.points.y)[:, np.newaxis]
    z = np.array(data.points.z)[:, np.newaxis]
    lidar_intensity = np.array(data.points.intensity, dtype=np.float32)[:, np.newaxis]
    lidar_pts = np.concatenate((x, y, z, lidar_intensity), axis=1)

    splitted_path = path.split('/')

    calib = read_json_file(path.replace(f'{splitted_path[-2]}/{splitted_path[-1]}', 'vehicle_calibration_info.json'))
    calib = get_calibration_config(calib, camera)
    cam2_to_velo = torch.from_numpy(calib.extrinsic)
    intrinsics = torch.tensor([calib.intrinsic[0, 0], calib.intrinsic[1, 1],
                               calib.intrinsic[0, 2], calib.intrinsic[1, 2]])

    return lidar_pts, cam2_to_velo.float(), intrinsics.float()


def get_scan_pandaset(path, sensor_id=0):
    scan = pd.read_pickle(path)
    scan = scan.loc[scan['d'] == sensor_id]

    return scan.values[:, :4]


def get_extrinsic_pandaset(camera):
    with open(os.path.join(os.path.dirname(__file__), 'pandaset_extrinsic.yaml')) as f:
        file_data = yaml.safe_load(f)
    camera_pose = file_data[camera]['extrinsic']['transform']
    camera_translation = torch.tensor([camera_pose['translation']['x'], camera_pose['translation']['y'],
                                       camera_pose['translation']['z']])
    camera_quaternion = torch.tensor([camera_pose['rotation']['w'], camera_pose['rotation']['x'],
                                      camera_pose['rotation']['y'], camera_pose['rotation']['z']])
    camera_pose = to_rotation_matrix(camera_quaternion, camera_translation)
    return camera_pose


class DatasetGeneralExtrinsicCalib(Dataset):

    def __init__(self, dataset_dirs, transform=None, augmentation=False, use_reflectance=False, max_t=2., max_r=10.,
                 train=True, normalize_images=True, dataset='kitti', cam='2', change_frame=False,
                 camera_intrinsics=None):
        super(DatasetGeneralExtrinsicCalib, self).__init__()
        self.dataset = dataset
        self.use_reflectance = use_reflectance
        self.max_r = max_r
        self.max_t = max_t
        self.augmentation = augmentation
        self.root_dirs = dataset_dirs
        self.transform = transform
        self.train = train
        self.normalize_images = normalize_images
        self.maps_folder = None
        self.extension = None
        self.cam = str(cam)
        self.camera_folder = f'image_{cam}'
        self.change_frame = change_frame
        if dataset == 'kitti':
            self.maps_folder = 'velodyne'
            self.extension = '.bin'
        elif dataset == 'argoverse':
            self.maps_folder = 'lidar'
            self.extension = '.ply'
            self.sdbs = {}
        elif dataset == 'custom':
            self.maps_folder = 'lidar'
            self.camera_folder = 'camera'
        self.all_files = []

        if not isinstance(dataset_dirs, list):
            dataset_dirs = [dataset_dirs]

        for directory in dataset_dirs:

            if dataset == 'argoverse':
                for log_id in sorted(os.listdir(directory)):
                    self.sdbs[log_id] = SynchronizationDB(directory, collect_single_log_id=log_id)

                    point_cloud_folder = os.path.join(directory, log_id, self.maps_folder)

                    sorted_filenames = sorted(os.listdir(point_cloud_folder))
                    for filename in sorted_filenames:
                        self.all_files.append(os.path.join(point_cloud_folder, filename))

            if dataset == 'custom':
                with open(os.path.join(directory, 'calibration.yaml')) as f:
                    file_data = yaml.safe_load(f)
                self.camera_intrinsics = torch.tensor(
                    [file_data['fx'], file_data['fy'], file_data['cx'], file_data['cy']])
                self.initial_extrinsic = torch.tensor(file_data['initial_extrinsic'], dtype=torch.float).reshape(4, 4)
                first_scan = os.listdir(os.path.join(directory, self.maps_folder))
                first_scan = sorted(first_scan)[0]
                self.extension = os.path.splitext(first_scan)[1]
                self.point_cloud_reader = _get_point_cloud_reader(self.extension[1:],
                                                                  os.path.join(directory, self.maps_folder, first_scan))

            if dataset == 'kitti' or dataset == 'custom':
                img_folder = os.path.join(directory, self.camera_folder)
                point_cloud_folder = os.path.join(directory, self.maps_folder)

                sorted_filenames = sorted(os.listdir(img_folder))
                for filename in sorted_filenames:
                    filename_no_extension = os.path.splitext(filename)[0]
                    point_cloud_path = os.path.join(point_cloud_folder, filename_no_extension + self.extension)
                    if not os.path.exists(point_cloud_path):
                        continue
                    self.all_files.append(os.path.join(img_folder, filename))

    def custom_transform(self, rgb, calib, img_rotation=0., flip=False):
        if self.train:
            color_transform = transforms.ColorJitter(0.2, 0.2, 0.2)
            rgb = color_transform(rgb)
        rgb = np.array(rgb)
        if self.train:
            if flip:
                rgb = cv2.flip(rgb, 1)
            height, width = rgb.shape[:2]
            matrix = cv2.getRotationMatrix2D(tuple(calib[2:].numpy()), img_rotation, 1.0)
            rgb = cv2.warpAffine(rgb, matrix, dsize=(width, height))

        return torch.tensor(rgb).float()

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        if self.dataset == 'kitti' or self.dataset == 'custom':
            img_path = self.all_files[idx]
            extension = os.path.basename(img_path)
            extension = os.path.splitext(extension)[1]
            pc_path = img_path.replace(f'/{self.camera_folder}/', f'/{self.maps_folder}/').replace(extension,
                                                                                                   self.extension)
        elif self.dataset == 'argoverse':
            pc_path = self.all_files[idx]

            splitted_path = pc_path.split('/')
            lidar_stamp = int(splitted_path[-1][3:-4])
            log_id = splitted_path[-3]
            sdb = self.sdbs[log_id]

        if self.dataset == 'kitti':
            pc, cam2vel, calib = get_scan_kitti(pc_path, cam=self.cam)
        elif self.dataset == 'argoverse':
            cam_timestamp = sdb.get_closest_cam_channel_timestamp(lidar_stamp, self.cam, log_id)

            img_path = pc_path.replace('/' + self.maps_folder + '/', f'/{self.cam}/')
            img_path = img_path.replace(splitted_path[-1], f'{self.cam}_{cam_timestamp}.jpg')

            pc, cam2vel, calib = get_scan_argo(pc_path, self.cam)
        elif self.dataset == 'custom':
            pc = self.point_cloud_reader(pc_path)
            cam2vel = self.initial_extrinsic
            calib = self.camera_intrinsics
            if pc.shape[1] == 3:
                pc = np.concatenate((pc, np.ones((pc.shape[0], 1))), 1)
            elif pc.shape[1] >= 4:
                pc = pc[:, :4]
            else:
                print("[ERROR], Point cloud has less than 3 channels")
                sys.exit(1)

        if self.use_reflectance:
            reflectance = torch.from_numpy(pc[:, -1]).float()
        pc[:, -1] = 1

        pc_in = torch.from_numpy(pc.astype(np.float32))
        pc_in = torch.mm(cam2vel, pc_in.t())
        if self.change_frame:
            pc_in = pc_in[[2, 0, 1, 3], :]

        try:
            img = Image.open(img_path).convert('RGB')
        except OSError as exc:
            return _retry_with_different_sample(self, idx, exc, img_path)
        h_mirror = False
        if np.random.rand() > 0.5 and self.train:
            h_mirror = True
            if self.change_frame:
                pc_in[1, :] *= -1
            else:
                pc_in[0, :] *= -1
            calib[2] = img.size[0] - calib[2]

        img_rotation = 0.
        if self.train:
            img_rotation = np.random.uniform(-5, 5)
        try:
            img = self.custom_transform(img, calib, img_rotation, h_mirror)
        except OSError as exc:
            return _retry_with_different_sample(self, idx, exc, img_path)

        # Rotate PointCloud for img_rotation
        if self.train:
            if self.change_frame:
                R = mathutils.Euler((radians(img_rotation), 0, 0), 'XYZ')
            else:
                R = mathutils.Euler((0, 0, radians(img_rotation)), 'XYZ')
            T = mathutils.Vector((0., 0., 0.))
            pc_in = rotate_forward(pc_in, R, T)

        max_angle = self.max_r
        rotz = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
        roty = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
        rotx = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
        transl_x = np.random.uniform(-self.max_t, self.max_t)
        transl_y = np.random.uniform(-self.max_t, self.max_t)
        transl_z = np.random.uniform(-self.max_t, min(self.max_t, 1.))

        if self.change_frame:
            R = mathutils.Euler((rotx, roty, rotz), 'XYZ')
            T = mathutils.Vector((transl_x, transl_y, transl_z))
        else:
            R = mathutils.Euler((roty, rotz, rotx), 'XYZ')
            T = mathutils.Vector((transl_y, transl_z, transl_x))


        R, T = invert_pose(R, T)
        R, T = torch.tensor(R), torch.tensor(T)

        sample = {'rgb': img, 'point_cloud': pc_in, 'calib': calib, 'tr_error': T,
                  'rot_error': R, 'rgb_name': img_path, 'idx': idx, 'cam2vel': cam2vel}
        if self.use_reflectance:
            sample['reflectance'] = reflectance

        return sample


class DatasetHerculesRadarExtrinsicCalib(Dataset):
    image_resize_scale = 0.5

    def __init__(self, dataset_dirs, transform=None, augmentation=False, use_reflectance=False, max_t=1.5,
                 max_r=20., train=True, normalize_images=True, change_frame=False, val_scene=None):
        super(DatasetHerculesRadarExtrinsicCalib, self).__init__()
        if isinstance(dataset_dirs, list):
            if len(dataset_dirs) != 1:
                raise ValueError("Hercules dataset expects a single dataset root directory.")
            dataset_dir = dataset_dirs[0]
        else:
            dataset_dir = dataset_dirs

        self.dataset = 'hercules'
        self.use_reflectance = use_reflectance
        self.max_r = max_r
        self.max_t = max_t
        self.augmentation = augmentation
        self.root_dir = dataset_dir
        self.transform = transform
        self.train = train
        self.normalize_images = normalize_images
        self.change_frame = change_frame
        self.val_scene = [val_scene] if isinstance(val_scene, str) else val_scene

        self.GTs_T_cam_radar = {}
        self.K = {}
        self.scene_data_dirs = {}
        self.scene_layouts = {}
        self.all_files = []
        self.root_calib_file = self._resolve_root_calibration_file()

        scene_list = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
        scene_list.sort()

        for scene in scene_list:
            scene_path = os.path.join(dataset_dir, scene)
            data_dir = self._resolve_scene_data_dir(scene_path)
            if data_dir is not None:
                self.scene_data_dirs[scene] = data_dir

        if self.val_scene is None:
            self.val_scene = [list(self.scene_data_dirs.keys())[0]] if self.scene_data_dirs else []

        for scene, data_dir in self.scene_data_dirs.items():
            calib_file = self.root_calib_file or self._resolve_scene_calibration_file(data_dir)
            if not os.path.exists(calib_file):
                continue

            with open(calib_file, 'r') as f:
                calib_data = yaml.safe_load(f)

            K_matrix = self._extract_intrinsic(calib_data)
            T_cam_radar = self._extract_extrinsic(calib_data)
            if K_matrix is None or T_cam_radar is None:
                continue

            if self.image_resize_scale != 1.0:
                K_matrix = K_matrix.copy()
                K_matrix[0, 0] *= self.image_resize_scale
                K_matrix[1, 1] *= self.image_resize_scale
                K_matrix[0, 2] *= self.image_resize_scale
                K_matrix[1, 2] *= self.image_resize_scale

            layout = self._resolve_scene_layout(os.path.join(dataset_dir, scene), data_dir)
            if layout is None:
                continue

            self.K[scene] = K_matrix
            self.GTs_T_cam_radar[scene] = T_cam_radar
            self.scene_layouts[scene] = layout

            include_scene = scene not in self.val_scene if self.train else scene in self.val_scene
            if not include_scene:
                continue

            if layout['pair_file'] is not None:
                scene_pairs = _load_sync_pairs(layout['pair_file'], layout['camera_dir'], layout['radar_dir'])
                for pair in scene_pairs:
                    pair['scene'] = scene
                    self.all_files.append(pair)
            else:
                image_list = [f for f in os.listdir(layout['camera_dir']) if is_image(f)]
                image_list.sort()
                radar_names = {
                    os.path.splitext(f)[0]: f
                    for f in _list_point_cloud_files(layout['radar_dir'])
                }
                for image_name in image_list:
                    base_name = os.path.splitext(image_name)[0]
                    if base_name not in radar_names:
                        continue
                    self.all_files.append({
                        'stamp': base_name,
                        'image_path': os.path.join(layout['camera_dir'], image_name),
                        'sensor_path': os.path.join(layout['radar_dir'], radar_names[base_name]),
                        'image_name': image_name,
                        'scene': scene,
                    })

        first_scan_file = None
        first_scan_ext = None
        for layout in self.scene_layouts.values():
            radar_files = _list_point_cloud_files(layout['radar_dir'])
            if radar_files:
                first_scan_file = os.path.join(layout['radar_dir'], radar_files[0])
                first_scan_ext = os.path.splitext(radar_files[0])[1].lstrip('.').lower()
                break
        if first_scan_file is None:
            raise RuntimeError(f"No radar point clouds found under {dataset_dir}")
        self.point_cloud_reader = _get_point_cloud_reader(first_scan_ext, first_scan_file)

    def _resolve_root_calibration_file(self):
        candidates = [
            os.path.join(self.root_dir, 'calibration.yaml'),
            os.path.join(self.root_dir, 'rlc_calibration.yaml'),
        ]
        for calib_file in candidates:
            if os.path.exists(calib_file):
                return calib_file
        return None

    def _resolve_scene_calibration_file(self, data_dir):
        candidates = [
            os.path.join(data_dir, 'calibration.yaml'),
            os.path.join(data_dir, 'rlc_calibration.yaml'),
        ]
        if os.path.basename(data_dir) != 'CMRNext':
            candidates.extend([
                os.path.join(data_dir, 'CMRNext', 'calibration.yaml'),
                os.path.join(data_dir, 'CMRNext', 'rlc_calibration.yaml'),
            ])
        for calib_file in candidates:
            if os.path.exists(calib_file):
                return calib_file
        return candidates[0]

    def _resolve_scene_data_dir(self, scene_path):
        candidates = [scene_path, os.path.join(scene_path, 'CMRNext')]
        for candidate in candidates:
            if not os.path.isdir(candidate):
                continue
            has_legacy_dirs = (
                os.path.isdir(os.path.join(candidate, 'camera')) and
                os.path.isdir(os.path.join(candidate, 'radar'))
            )
            has_pair_dirs = os.path.isdir(os.path.join(candidate, 'offline'))
            if has_legacy_dirs or has_pair_dirs:
                return candidate
        return None

    def _resolve_scene_layout(self, scene_path, data_dir):
        pair_filenames = [
            'image_Cam0_radar_Continental.txt',
            'image_left_radar_Continental.txt',
        ]
        pair_candidates = []
        for pair_name in pair_filenames:
            pair_candidates.extend([
                os.path.join(scene_path, 'offline', 'synced_stamps', pair_name),
                os.path.join(data_dir, 'offline', 'synced_stamps', pair_name),
                os.path.join(os.path.dirname(data_dir), 'offline', 'synced_stamps', pair_name),
            ])

        sensor_root_candidates = [
            os.path.join(scene_path, 'offline', 'sensor_data'),
            os.path.join(data_dir, 'offline', 'sensor_data'),
            os.path.join(os.path.dirname(data_dir), 'offline', 'sensor_data'),
        ]
        camera_dir_names = ['image_Cam0', 'image_left']
        radar_dir_names = ['radar_Continental', 'radar']

        for pair_file in pair_candidates:
            if not os.path.exists(pair_file):
                continue
            for sensor_root in sensor_root_candidates:
                for camera_dir_name in camera_dir_names:
                    for radar_dir_name in radar_dir_names:
                        camera_dir = os.path.join(sensor_root, camera_dir_name)
                        radar_dir = os.path.join(sensor_root, radar_dir_name)
                        if os.path.isdir(camera_dir) and os.path.isdir(radar_dir):
                            return {
                                'camera_dir': camera_dir,
                                'radar_dir': radar_dir,
                                'pair_file': pair_file,
                            }

        for camera_dir_name in ['camera', 'image_left']:
            for radar_dir_name in ['radar', 'radar_Continental']:
                camera_dir = os.path.join(data_dir, camera_dir_name)
                radar_dir = os.path.join(data_dir, radar_dir_name)
                if os.path.isdir(camera_dir) and os.path.isdir(radar_dir):
                    return {
                        'camera_dir': camera_dir,
                        'radar_dir': radar_dir,
                        'pair_file': None,
                    }
        return None

    def _extract_intrinsic(self, calib_data):
        K_matrix = None
        if calib_data and 'camera' in calib_data and 'intrinsic' in calib_data['camera']:
            K_matrix = np.array(calib_data['camera']['intrinsic'], dtype=np.float32)
            if K_matrix.shape != (3, 3):
                K_matrix = None
        if K_matrix is None and calib_data and all(key in calib_data for key in ['fx', 'fy', 'cx', 'cy']):
            K_matrix = np.array([
                [float(calib_data['fx']), 0.0, float(calib_data['cx'])],
                [0.0, float(calib_data['fy']), float(calib_data['cy'])],
                [0.0, 0.0, 1.0],
            ], dtype=np.float32)
        return K_matrix

    def _extract_extrinsic(self, calib_data):
        T_cam_radar = None
        if calib_data and 'extrinsic' in calib_data:
            if 'T_cam_radar' in calib_data['extrinsic']:
                T_cam_radar = np.array(calib_data['extrinsic']['T_cam_radar'], dtype=np.float32)
                if T_cam_radar.shape != (4, 4):
                    T_cam_radar = None
            elif 'rotation' in calib_data['extrinsic'] and 'translation' in calib_data['extrinsic']:
                rotation = np.array(calib_data['extrinsic']['rotation'], dtype=np.float32)
                translation = np.array(calib_data['extrinsic']['translation'], dtype=np.float32)
                if rotation.shape == (3, 3) and translation.shape == (3,):
                    T_cam_radar = np.eye(4, dtype=np.float32)
                    T_cam_radar[:3, :3] = rotation
                    T_cam_radar[:3, 3] = translation
        if T_cam_radar is None and calib_data and 'initial_extrinsic' in calib_data:
            ext_list = np.array(calib_data['initial_extrinsic'], dtype=np.float32).reshape(-1)
            if ext_list.size == 16:
                T_cam_radar = ext_list.reshape(4, 4)
        return T_cam_radar

    def custom_transform(self, rgb, calib, img_rotation=0., flip=False):
        if self.train:
            color_transform = transforms.ColorJitter(0.2, 0.2, 0.2)
            rgb = color_transform(rgb)
        rgb = np.array(rgb)
        if self.train:
            if flip:
                rgb = cv2.flip(rgb, 1)
            height, width = rgb.shape[:2]
            matrix = cv2.getRotationMatrix2D(tuple(calib[2:].numpy()), img_rotation, 1.0)
            rgb = cv2.warpAffine(rgb, matrix, dsize=(width, height))
        return torch.tensor(rgb).float()

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        item = self.all_files[idx]
        scene = item['scene']
        img_path = item['image_path']
        radar_path = item['sensor_path']

        pc = self.point_cloud_reader(radar_path)
        if pc.ndim == 1:
            pc = pc.reshape(1, -1)
        if pc.shape[1] < 3:
            raise RuntimeError(f"Radar point cloud has invalid shape: {pc.shape}")
        if pc.shape[1] == 3:
            pc = np.concatenate((pc, np.ones((pc.shape[0], 1), dtype=np.float32)), axis=1)
        else:
            pc = pc[:, :4]

        # Match the filtering used in the LCCNet Hercules radar loader.
        mask = ((pc[:, 0] < -3.) | (pc[:, 0] > 3.) | (pc[:, 1] < -3.) | (pc[:, 1] > 3.))
        pc = pc[mask]
        if pc.shape[0] == 0:
            raise RuntimeError(f"Radar point cloud became empty after filtering: {radar_path}")

        calib_np = self.K[scene]
        calib = torch.tensor([calib_np[0, 0], calib_np[1, 1], calib_np[0, 2], calib_np[1, 2]]).float()
        cam2vel = torch.from_numpy(self.GTs_T_cam_radar[scene]).float()

        if self.use_reflectance:
            reflectance = torch.from_numpy(pc[:, -1]).float()
        pc[:, -1] = 1.
        pc_in = torch.from_numpy(pc.astype(np.float32))
        pc_in = torch.mm(cam2vel, pc_in.t())
        if self.change_frame:
            pc_in = pc_in[[2, 0, 1, 3], :]

        try:
            img = Image.open(img_path).convert('RGB')
            if self.image_resize_scale != 1.0:
                resized_width = max(1, int(round(img.width * self.image_resize_scale)))
                resized_height = max(1, int(round(img.height * self.image_resize_scale)))
                img = img.resize((resized_width, resized_height), Image.BILINEAR)
        except OSError as exc:
            return _retry_with_different_sample(self, idx, exc, img_path)

        h_mirror = False
        if np.random.rand() > 0.5 and self.train:
            h_mirror = True
            if self.change_frame:
                pc_in[1, :] *= -1
            else:
                pc_in[0, :] *= -1
            calib[2] = img.size[0] - calib[2]

        img_rotation = np.random.uniform(-5, 5) if self.train else 0.
        try:
            img = self.custom_transform(img, calib, img_rotation, h_mirror)
        except OSError as exc:
            return _retry_with_different_sample(self, idx, exc, img_path)

        if self.train:
            if self.change_frame:
                R_img = mathutils.Euler((radians(img_rotation), 0, 0), 'XYZ')
            else:
                R_img = mathutils.Euler((0, 0, radians(img_rotation)), 'XYZ')
            T_img = mathutils.Vector((0., 0., 0.))
            pc_in = rotate_forward(pc_in, R_img, T_img)

        max_angle = self.max_r
        rotz = np.random.uniform(-max_angle, max_angle) * (np.pi / 180.0)
        roty = np.random.uniform(-max_angle, max_angle) * (np.pi / 180.0)
        rotx = np.random.uniform(-max_angle, max_angle) * (np.pi / 180.0)
        transl_x = np.random.uniform(-self.max_t, self.max_t)
        transl_y = np.random.uniform(-self.max_t, self.max_t)
        transl_z = np.random.uniform(-self.max_t, min(self.max_t, 1.0))

        if self.change_frame:
            R = mathutils.Euler((rotx, roty, rotz), 'XYZ')
            T = mathutils.Vector((transl_x, transl_y, transl_z))
        else:
            R = mathutils.Euler((roty, rotz, rotx), 'XYZ')
            T = mathutils.Vector((transl_y, transl_z, transl_x))

        R, T = invert_pose(R, T)
        R, T = torch.tensor(R), torch.tensor(T)

        sample = {
            'rgb': img,
            'point_cloud': pc_in,
            'calib': calib,
            'tr_error': T,
            'rot_error': R,
            'rgb_name': img_path,
            'idx': idx,
            'cam2vel': cam2vel,
        }
        if self.use_reflectance:
            sample['reflectance'] = reflectance
        return sample


class DatasetPandasetExtrinsicCalib(Dataset):

    def __init__(self, dataset_dirs, transform=None, augmentation=False, use_reflectance=False, max_t=2., max_r=10.,
                 train=True, normalize_images=True, sensor_id=0, camera='front_camera', change_frame=False):
        super(DatasetPandasetExtrinsicCalib, self).__init__()
        self.use_reflectance = use_reflectance
        self.max_r = max_r
        self.max_t = max_t
        self.augmentation = augmentation
        self.root_dirs = dataset_dirs
        self.transform = transform
        self.train = train
        self.normalize_images = normalize_images
        self.sensor_id = sensor_id
        self.maps_folder = 'lidar'
        self.extension = 'pkl.gz'
        self.camera = camera

        self.all_files = []
        self.camera_poses = []
        self.camera_stamps = []

        self.change_frame = change_frame

        if not isinstance(dataset_dirs, list):
            dataset_dirs = [dataset_dirs]

        for directory in dataset_dirs:
            point_cloud_folder = os.path.join(directory, self.maps_folder)

            sorted_filenames = sorted(os.listdir(point_cloud_folder))
            for filename in sorted_filenames:
                if '.json' not in filename:
                    self.all_files.append(os.path.join(point_cloud_folder, filename))

            pose_file = os.path.join(directory, 'camera', camera, 'poses.json')
            timestamp_file = os.path.join(directory, 'camera', camera, 'timestamps.json')
            with open(pose_file, 'r') as f:
                file_data = json.load(f)
                for entry in file_data:
                    self.camera_poses.append(_heading_position_to_mat(entry['heading'], entry['position']))

            with open(timestamp_file, 'r') as f:
                file_data = json.load(f)
                for entry in file_data:
                    self.camera_stamps.append(entry)

    def custom_transform(self, rgb, calib, img_rotation=0., flip=False):
        if self.train:
            color_transform = transforms.ColorJitter(0.2, 0.2, 0.2)
            rgb = color_transform(rgb)
        rgb = np.array(rgb)
        if self.train:
            if flip:
                rgb = cv2.flip(rgb, 1)
            height, width = rgb.shape[:2]
            matrix = cv2.getRotationMatrix2D(tuple(calib[2:].numpy()), img_rotation, 1.0)
            rgb = cv2.warpAffine(rgb, matrix, dsize=(width, height))

        return torch.tensor(rgb).float()

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        pc_path = self.all_files[idx]
        img_path = pc_path.replace('/' + self.maps_folder + '/', f'/camera/{self.camera}/').replace(self.extension,
                                                                                                    'jpg')

        # Get the camera intrinsic parameters
        calib_file = os.path.dirname(img_path)
        calib_file = os.path.join(calib_file, 'intrinsics.json')
        with open(calib_file, 'r') as f:
            calib = json.load(f)
        calib = torch.tensor([calib['fx'], calib['fy'], calib['cx'], calib['cy']]).float()

        pc = get_scan_pandaset(pc_path, self.sensor_id)
        if self.use_reflectance:
            reflectance = torch.from_numpy(pc[:, -1]).float()
        pc[:, -1] = 1
        cam_pose = torch.from_numpy(self.camera_poses[idx]).float().inverse()
        pc_in = torch.from_numpy(pc.astype(np.float32))
        pc_in = torch.mm(cam_pose, pc_in.t())
        if self.change_frame:
            pc_in = pc_in[[2, 0, 1, 3], :]

        cam2vel = get_extrinsic_pandaset(self.camera)

        try:
            img = Image.open(img_path).convert('RGB')
        except OSError as exc:
            return _retry_with_different_sample(self, idx, exc, img_path)
        h_mirror = False
        if np.random.rand() > 0.5 and self.train:
            h_mirror = True
            if self.change_frame:
                pc_in[1, :] *= -1
            else:
                pc_in[0, :] *= -1
            calib[2] = img.size[0] - calib[2]

        img_rotation = 0.
        if self.train:
            img_rotation = np.random.uniform(-5, 5)
        try:
            img = self.custom_transform(img, calib, img_rotation, h_mirror)
        except OSError as exc:
            return _retry_with_different_sample(self, idx, exc, img_path)

        # Rotate PointCloud for img_rotation
        if self.train:
            if self.change_frame:
                R = mathutils.Euler((radians(img_rotation), 0, 0), 'XYZ')
            else:
                R = mathutils.Euler((0, 0, radians(img_rotation)), 'XYZ')
            T = mathutils.Vector((0., 0., 0.))
            pc_in = rotate_forward(pc_in, R, T)

        max_angle = self.max_r
        rotz = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
        roty = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
        rotx = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
        transl_x = np.random.uniform(-self.max_t, self.max_t)
        transl_y = np.random.uniform(-self.max_t, self.max_t)
        transl_z = np.random.uniform(-self.max_t, min(self.max_t, 1.))

        if self.change_frame:
            R = mathutils.Euler((rotx, roty, rotz), 'XYZ')
            T = mathutils.Vector((transl_x, transl_y, transl_z))
        else:
            R = mathutils.Euler((roty, rotz, rotx), 'XYZ')
            T = mathutils.Vector((transl_y, transl_z, transl_x))

        R, T = invert_pose(R, T)
        R, T = torch.tensor(R), torch.tensor(T)

        sample = {'rgb': img, 'point_cloud': pc_in, 'calib': calib, 'tr_error': T,
                  'rot_error': R, 'rgb_name': img_path, 'idx': idx, 'cam2vel': cam2vel}
        if self.use_reflectance:
            sample['reflectance'] = reflectance

        return sample
