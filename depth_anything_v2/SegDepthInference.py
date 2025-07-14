import argparse
import numpy as np
from scipy.ndimage import binary_dilation
from PIL import Image
import cv2
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from scipy.stats import spearmanr
from enum import Enum
from time import time

import open3d as o3d

from depth_metrics import comprehensive_depth_evaluation
from seg_deformable_depth import SegmentationDeformableDepth
from util.vis import decode_segmap

class SegLabels(Enum):
    BACKGROUND = 0
    HEAD = 1
    BASEPLATE = 2
    PREVIOUS_PART = 3
    CURRENT_PART = 4
    WELD_FLASH = 5

def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation and Depth Inference')
    parser.add_argument('--device', default='cuda', help='Training device')
    parser.add_argument('--image-path', type=str, required=True, help='Path to input image')
    parser.add_argument('--model-weights-dir', type=str, default="../model_weights",
                        help='pretrained model weights directory')
    parser.add_argument('--class-a', type=int, default=SegLabels.BASEPLATE.value, help='Class A ID for relative depth comparison')
    parser.add_argument('--class-b', type=int, default=SegLabels.CURRENT_PART.value, help='Class B ID for relative depth comparison')
    parser.add_argument('--rim-width', type=int, default=20, help='Ring width (pixels) around part for depth contrast')
    parser.add_argument('--interactive', action='store_true', default=False, help='Show interactive 3D pointcloud viewer (Open3D)')
    return parser.parse_args()

def make_divisible(val, divisor=14):
    if isinstance(val, (tuple, list, np.ndarray)):
        return tuple(v - (v % divisor) for v in val)
    else:
        return val - (val % divisor)

def pixel_to_3d_coordinates(xs, ys, depths, fx, fy, cx, cy):
    """
    Convert pixel coordinates and depths to 3D world coordinates.

    Args:
        xs, ys: pixel coordinates (arrays)
        depths: depth values at those pixels (array)
        fx, fy: focal lengths in pixels
        cx, cy: principal point coordinates (image center)

    Returns:
        3D points as (N, 3) array in world coordinates
    """
    # Convert to normalized camera coordinates
    x_3d = (xs - cx) * depths / fx
    y_3d = (ys - cy) * depths / fy
    z_3d = depths

    return np.stack([x_3d, y_3d, z_3d], axis=1).astype(np.float32)

def estimate_camera_intrinsics(img_width, img_height, fov_degrees=60):
    """
    Estimate camera intrinsics assuming a typical field of view.

    Args:
        img_width, img_height: image dimensions
        fov_degrees: horizontal field of view in degrees

    Returns:
        fx, fy, cx, cy: camera intrinsic parameters
    """
    # Convert FOV to radians
    fov_rad = np.radians(fov_degrees)

    # Calculate focal length from FOV
    fx = img_width / (2 * np.tan(fov_rad / 2))
    fy = fx  # Assuming square pixels

    # Principal point at image center
    cx = img_width / 2
    cy = img_height / 2

    return fx, fy, cx, cy


def set_consistent_view(vis_control):
    """Set a consistent view for both renderers"""
    vis_control.set_up([0, -1.0, 0])  # Y is downwards
    vis_control.set_front([0.5, 0.5, -1.0])  # View direction
    vis_control.set_lookat([0, 0, 0])  # Look at origin
    vis_control.set_zoom(0.8)  # Zoom level


def capture_pointcloud_image_with_rim_fixed(
        rgb_img, depth_map, seg_map, class_id, rim_mask, max_points=100000, rim_points=50000, img_size=600
):
    """
    Renders the 3D pointcloud as an image using Open3D's offscreen renderer with proper 3D coordinates.
    Returns the rendered image and the camera view parameters for consistency.
    """
    mask = seg_map == class_id
    if not np.any(mask):
        print(f"No pixels found for class {class_id}")
        return np.zeros((img_size, img_size, 3), dtype=np.uint8)

    # Get image dimensions
    img_height, img_width = depth_map.shape

    # Estimate camera intrinsics
    fx, fy, cx, cy = estimate_camera_intrinsics(img_width, img_height)

    # Get pixel coordinates and depths for the main object
    ys, xs = np.where(mask)
    zs = depth_map[mask]
    img_arr = np.array(rgb_img)
    colors = img_arr[mask].astype(np.float32) / 255.0

    # Subsample for performance
    if len(xs) > max_points:
        idx = np.random.choice(len(xs), max_points, replace=False)
        xs, ys, zs, colors = xs[idx], ys[idx], zs[idx], colors[idx]

    # Convert to 3D coordinates
    part_points = pixel_to_3d_coordinates(xs, ys, zs, fx, fy, cx, cy)

    # Handle rim points
    rim_ys, rim_xs = np.where(rim_mask)
    rim_zs = depth_map[rim_mask]
    rim_colors = img_arr[rim_mask].astype(np.float32) / 255.0
    cyan = np.array([0.0, 1.0, 1.0], dtype=np.float32)
    alpha = 0.3
    rim_colors = alpha * cyan + (1 - alpha) * rim_colors

    if len(rim_xs) > rim_points:
        idx = np.random.choice(len(rim_xs), rim_points, replace=False)
        rim_xs, rim_ys, rim_zs, rim_colors = rim_xs[idx], rim_ys[idx], rim_zs[idx], rim_colors[idx]

    # Convert rim points to 3D coordinates
    rim_points_xyz = pixel_to_3d_coordinates(rim_xs, rim_ys, rim_zs, fx, fy, cx, cy)

    # Center and scale (combine both clouds before scaling)
    all_points = np.concatenate([part_points, rim_points_xyz], axis=0)
    centroid = np.mean(all_points, axis=0)
    all_points -= centroid
    scale = np.max(np.linalg.norm(all_points, axis=1))
    all_points /= (scale + 1e-6)
    all_points *= img_size * 0.4

    # Split back to part/rim
    n_part = part_points.shape[0]
    part_points_scaled = all_points[:n_part]
    rim_points_scaled = all_points[n_part:]

    # Create Open3D geometries
    pcd_part = o3d.geometry.PointCloud()
    pcd_part.points = o3d.utility.Vector3dVector(part_points_scaled)
    pcd_part.colors = o3d.utility.Vector3dVector(colors)

    pcd_rim = o3d.geometry.PointCloud()
    pcd_rim.points = o3d.utility.Vector3dVector(rim_points_scaled)
    pcd_rim.colors = o3d.utility.Vector3dVector(rim_colors)

    # Render both clouds
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=img_size, height=img_size)
    vis.add_geometry(pcd_part)
    vis.add_geometry(pcd_rim)

    ctr = vis.get_view_control()
    set_consistent_view(ctr)

    vis.update_geometry(pcd_part)
    vis.update_geometry(pcd_rim)
    vis.poll_events()
    vis.update_renderer()
    img = vis.capture_screen_float_buffer(do_render=True)
    # Get camera parameters for later use
    camera_params = ctr.convert_to_pinhole_camera_parameters()

    vis.destroy_window()
    img = (255 * np.asarray(img)).astype(np.uint8)
    return img, camera_params


# Also fix the interactive visualization section
def create_interactive_pointcloud_fixed(cropped_raw_img, depth_map, seg_map, rim_mask, class_b, SegLabels,
                                        camera_params=None):
    """
    Create interactive 3D point cloud with proper coordinate transformation.
    If camera_params is provided, use the same view as the offscreen renderer.
    """
    mask = seg_map == class_b
    img_height, img_width = depth_map.shape
    fx, fy, cx, cy = estimate_camera_intrinsics(img_width, img_height)

    img_arr = np.array(cropped_raw_img)
    ys, xs = np.where(mask)
    zs = depth_map[mask]
    colors = img_arr[mask].astype(np.float32) / 255.0

    # Convert to 3D coordinates
    part_points = pixel_to_3d_coordinates(xs, ys, zs, fx, fy, cx, cy)

    # Handle rim points
    rim_ys, rim_xs = np.where(rim_mask)
    rim_zs = depth_map[rim_mask]
    rim_colors = img_arr[rim_mask].astype(np.float32) / 255.0
    cyan = np.array([0.0, 1.0, 1.0], dtype=np.float32)
    alpha = 0.3
    rim_colors = alpha * cyan + (1 - alpha) * rim_colors

    # Convert rim points to 3D coordinates
    rim_points_xyz = pixel_to_3d_coordinates(rim_xs, rim_ys, rim_zs, fx, fy, cx, cy)

    # Center and scale
    all_points = np.concatenate([part_points, rim_points_xyz], axis=0)
    centroid = np.mean(all_points, axis=0)
    all_points -= centroid
    scale = np.max(np.linalg.norm(all_points, axis=1))
    all_points /= (scale + 1e-6)
    all_points *= 600 * 0.4

    n_part = part_points.shape[0]
    part_points_scaled = all_points[:n_part]
    rim_points_scaled = all_points[n_part:]

    pcd_part = o3d.geometry.PointCloud()
    pcd_part.points = o3d.utility.Vector3dVector(part_points_scaled)
    pcd_part.colors = o3d.utility.Vector3dVector(colors)

    pcd_rim = o3d.geometry.PointCloud()
    pcd_rim.points = o3d.utility.Vector3dVector(rim_points_scaled)
    pcd_rim.colors = o3d.utility.Vector3dVector(rim_colors)

    # Create interactive visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Interactive 3D Pointcloud (class {SegLabels(class_b).name})")
    vis.add_geometry(pcd_part)
    vis.add_geometry(pcd_rim)

    # Set the same camera view as the offscreen renderer if provided
    if camera_params is not None:
        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(camera_params, allow_arbitrary=True)
    else:
        # Use default view settings
        ctr = vis.get_view_control()
        set_consistent_view(ctr)


    vis.run()
    vis.destroy_window()

def plot_full_results(raw_img, seg_map_color, depth_map_vis, pc_img, seglabels, class_id):
    """
    Shows a 2x2 grid:
      [Original, Segmentation]
      [Depth,    Pointcloud (with rim)]
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    axs[0, 0].imshow(raw_img)
    axs[0, 0].set_title("Original Image")
    axs[0, 0].axis('off')
    axs[0, 1].imshow(seg_map_color)
    axs[0, 1].set_title("Predicted Segmentation")
    axs[0, 1].axis('off')
    axs[1, 0].imshow(depth_map_vis)
    axs[1, 0].set_title("Predicted Depth")
    axs[1, 0].axis('off')
    axs[1, 1].imshow(pc_img)
    axs[1, 1].set_title(f"3D Pointcloud of: {seglabels(class_id).name}")
    axs[1, 1].axis('off')
    plt.tight_layout()
    plt.show()

def local_rim_mask(seg_map, class_A, class_B, rim_width=10):
    """
    Returns a binary mask (H, W) of rim region around B that is within class A.
    """
    mask_A = (seg_map == class_A)
    mask_B = (seg_map == class_B)
    dilated_B = binary_dilation(mask_B, iterations=rim_width)
    rim_mask = np.logical_and(dilated_B, mask_A)
    return rim_mask

def local_rim_depth_diff(seg_map, depth, class_A, class_B, SegLabels, rim_width=5):
    mask_A = (seg_map == class_A)
    mask_B = (seg_map == class_B)
    if not (np.any(mask_A) and np.any(mask_B)):
        print(f"\nCannot compute relative depth: one or both classes not present in segmentation.")
        return
    mask_B_2d = mask_B
    mask_A_2d = mask_A
    depth_2d = depth
    dilated_B = binary_dilation(mask_B_2d, iterations=rim_width)
    rim_A = np.logical_and(dilated_B, mask_A_2d)
    by, bx = np.where(mask_B_2d)
    ay, ax = np.where(rim_A)
    if len(ay) == 0:
        print("No rim found, try increasing rim_width.")
        return
    A_coords = np.stack([ay, ax], axis=1)
    B_coords = np.stack([by, bx], axis=1)
    if len(B_coords) > 1000:
        idx = np.random.choice(len(B_coords), 1000, replace=False)
        B_coords = B_coords[idx]
    dists = np.linalg.norm(B_coords[:, None, :] - A_coords[None, :, :], axis=2)
    idx_closest = np.argmin(dists, axis=1)
    A_closest = A_coords[idx_closest]
    d_B = depth_2d[B_coords[:, 0], B_coords[:, 1]]
    d_A = depth_2d[A_closest[:, 0], A_closest[:, 1]]
    avg_diff = np.mean(d_B - d_A)
    print(f"\n--- Relative Depth (rim method) ---")
    print(f"Avg. Depth ({SegLabels(class_B).name}) - Local Rim Depth ({SegLabels(class_A).name}) = {avg_diff:.4f}")

if __name__ == '__main__':
    args = parse_args()
    raw_img = Image.open(args.image_path).convert('RGB')
    img_w, img_h = make_divisible(raw_img.size)

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = SegmentationDeformableDepth(
        encoder="vitb",
        num_classes=6,
        image_height=img_h,
        image_width=img_w,
        features=768,
        out_channels=[256, 512, 1024, 1024],
        model_weights_dir=args.model_weights_dir,
        device=device,
    )
    model.eval().to(device)

    input_transform = transforms.Compose([
        transforms.CenterCrop((img_h, img_w)),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    raw_img_transform = transforms.Compose([
        transforms.CenterCrop((img_h, img_w)),
    ])
    transformed_image = input_transform(raw_img).unsqueeze(0).to(device)
    start_time = time()
    with torch.no_grad():
        depth, seg_map = model.infer_image(transformed_image)
    end_time = time()
    print(f"SegDepthInference took {end_time - start_time:.2f} seconds")

    # Process segmentation output
    seg_map = seg_map[0]
    seg_map_color = decode_segmap(seg_map)  # (H, W, 3)

    # Process depth output
    depth_map = depth[0]
    print(f"Depth Range: Max {np.max(depth_map)}, Min: {np.min(depth_map)}")
    depth_map_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
    depth_map_inv = 1.0 - depth_map_norm
    depth_map_vis = cv2.applyColorMap((depth_map_inv * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)

    cropped_raw_img = raw_img_transform(raw_img)

    # --- Rim mask for the plot and metrics
    rim_mask = local_rim_mask(seg_map, args.class_a, args.class_b, rim_width=args.rim_width)

    # --- Render pointcloud view to image for the selected object (with rim)
    pc_img, camera_params = capture_pointcloud_image_with_rim_fixed(
        cropped_raw_img, depth_map, seg_map, class_id=args.class_b, rim_mask=rim_mask, img_size=600
    )

    # --- Show all results together
    plot_full_results(
        cropped_raw_img,
        seg_map_color,
        depth_map_vis,
        pc_img,
        SegLabels,
        class_id=args.class_b
    )

    if args.interactive:
        create_interactive_pointcloud_fixed(
            cropped_raw_img, depth_map, seg_map, rim_mask, args.class_b, SegLabels, camera_params
        )

    # ---- Metrics ----
    # Relative Depth: Class B vs Class A
    # local_rim_depth_diff(seg_map, depth_map, args.class_a, args.class_b, SegLabels, rim_width=args.rim_width)
    comprehensive_depth_evaluation(depth_map, seg_map, class_labels_enum=SegLabels)