import argparse
import os

import cv2
import numpy as np
import torch
from PIL import Image
from matplotlib import cm

from camera_model import CameraModel
from datasets.DatasetExtrinsicCalib import DatasetHerculesRadarExtrinsicCalib


def infer_hercules_img_shape(dataset_root, val_scene=None):
    probe_dataset = DatasetHerculesRadarExtrinsicCalib(dataset_root, train=False, val_scene=val_scene)
    if len(probe_dataset) == 0:
        raise RuntimeError(f"No Hercules validation samples found under {dataset_root}")

    first_image = Image.open(probe_dataset.all_files[0]['image_path']).convert('RGB')
    width = max(1, int(round(first_image.width * probe_dataset.image_resize_scale)))
    height = max(1, int(round(first_image.height * probe_dataset.image_resize_scale)))
    height = 64 * ((height + 63) // 64)
    width = 64 * ((width + 63) // 64)
    return [height, width]


def pad_bottom_right(image, target_shape, pad_value=0):
    height, width = image.shape[:2]
    target_height, target_width = target_shape
    if height >= target_height and width >= target_width:
        return image

    pad_bottom = max(0, target_height - height)
    pad_right = max(0, target_width - width)
    return cv2.copyMakeBorder(
        image,
        0,
        pad_bottom,
        0,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=(pad_value, pad_value, pad_value),
    )


def colorize_depth(depth):
    if depth.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)

    depth = depth.astype(np.float32)
    depth_min = float(depth.min())
    depth_max = float(depth.max())
    if depth_max - depth_min < 1e-6:
        normalized = np.zeros_like(depth)
    else:
        normalized = (depth - depth_min) / (depth_max - depth_min)
    colors = cm.get_cmap('turbo')(normalized)[:, :3]
    return (colors[:, ::-1] * 255.0).astype(np.uint8)


def compute_visible_points(uv, depth, image_height, image_width):
    if uv.shape[0] == 0:
        return np.zeros((0,), dtype=bool)

    visible = np.zeros((uv.shape[0],), dtype=bool)
    z_buffer = np.full((image_height, image_width), np.inf, dtype=np.float32)
    selected_indices = np.full((image_height, image_width), -1, dtype=np.int64)

    depth_np = depth.astype(np.float32)
    order = np.argsort(depth_np)
    for point_idx in order:
        u, v = uv[point_idx]
        current_depth = depth_np[point_idx]
        if current_depth < z_buffer[v, u]:
            previous_idx = selected_indices[v, u]
            if previous_idx >= 0:
                visible[previous_idx] = False
            z_buffer[v, u] = current_depth
            selected_indices[v, u] = point_idx
            visible[point_idx] = True

    return visible


def project_visible_points(sample):
    rgb = sample['rgb']
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.cpu().numpy()
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    point_cloud = sample['point_cloud']
    if not isinstance(point_cloud, torch.Tensor):
        point_cloud = torch.as_tensor(point_cloud)
    point_cloud = point_cloud.float().cpu()

    calib = sample['calib']
    if not isinstance(calib, torch.Tensor):
        calib = torch.as_tensor(calib)
    calib = calib.float().cpu()

    cam_model = CameraModel()
    cam_model.focal_length = calib[:2]
    cam_model.principal_point = calib[2:]

    image_shape = [rgb.shape[0], rgb.shape[1], rgb.shape[2]]
    uv, depth, _, _ = cam_model.project_pytorch(point_cloud.clone(), image_shape)
    uv_np = uv.t().int().cpu().numpy()
    depth_np = depth.cpu().numpy()
    visible_mask = compute_visible_points(uv_np, depth_np, rgb.shape[0], rgb.shape[1])
    uv_visible = uv_np[visible_mask]
    depth_visible = depth_np[visible_mask]
    return rgb, uv_visible, depth_visible


def draw_projection(rgb, uv, depth, point_radius):
    overlay = rgb.copy()
    colors = colorize_depth(depth)
    for (u, v), color in zip(uv, colors):
        cv2.circle(overlay, (int(u), int(v)), point_radius, color.tolist(), -1, lineType=cv2.LINE_AA)
    blended = cv2.addWeighted(overlay, 0.75, rgb, 0.25, 0.0)
    return blended


def build_output_path(output_dir, split, scene, stamp):
    scene_dir = os.path.join(output_dir, split, scene)
    os.makedirs(scene_dir, exist_ok=True)
    return os.path.join(scene_dir, f"{stamp}.png")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize Hercules radar points projected onto images using the ground-truth extrinsic."
    )
    parser.add_argument('--data_folder_hercules', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--val_scene', type=str, default='library_1')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'])
    parser.add_argument('--max_t', type=float, default=1.5)
    parser.add_argument('--max_r', type=float, default=20.0)
    parser.add_argument('--max_samples', type=int, default=0,
                        help='0 means all samples in the selected split.')
    parser.add_argument('--num_projections', type=int, default=None,
                        help='Number of projection images to save. Overrides --max_samples when set.')
    parser.add_argument('--point_radius', type=int, default=2)
    parser.add_argument('--pad_to_train_shape', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()

    split_is_train = args.split == 'train'
    dataset = DatasetHerculesRadarExtrinsicCalib(
        args.data_folder_hercules,
        train=split_is_train,
        max_r=args.max_r,
        max_t=args.max_t,
        use_reflectance=False,
        normalize_images=True,
        val_scene=args.val_scene,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No Hercules samples found for split={args.split} under {args.data_folder_hercules}")

    padded_shape = None
    if args.pad_to_train_shape:
        padded_shape = infer_hercules_img_shape(args.data_folder_hercules, args.val_scene)

    requested_count = args.num_projections if args.num_projections is not None else args.max_samples
    total = len(dataset) if requested_count <= 0 else min(len(dataset), requested_count)
    print(f"Saving {total} projected images to {args.output_dir}")

    for idx in range(total):
        sample = dataset[idx]
        meta = dataset.all_files[idx]
        rgb, uv, depth = project_visible_points(sample)
        projection = draw_projection(rgb, uv, depth, args.point_radius)
        if padded_shape is not None:
            projection = pad_bottom_right(projection, padded_shape)

        stamp = meta.get('stamp', os.path.splitext(meta.get('image_name', f"{idx:06d}"))[0])
        output_path = build_output_path(args.output_dir, args.split, meta['scene'], stamp)
        cv2.imwrite(output_path, cv2.cvtColor(projection, cv2.COLOR_RGB2BGR))

        if (idx + 1) % 50 == 0 or idx + 1 == total:
            print(f"[{idx + 1}/{total}] saved {output_path}")


if __name__ == '__main__':
    main()
