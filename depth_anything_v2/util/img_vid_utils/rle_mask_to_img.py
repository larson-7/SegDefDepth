import argparse
import os
import json
from collections import OrderedDict
import cv2
import numpy as np
from tqdm import tqdm
import pycocotools.mask as mask_utils
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Convert RLE masks to images with overlays.")
    parser.add_argument('--data-dir', type=str, default=None, help='Base data directory')
    parser.add_argument('--output-images', type=str, default=None, help='Output RGB images directory')
    parser.add_argument('--output-masks', type=str, default=None, help='Output masks directory')
    parser.add_argument('--output-masks-color', type=str, default=None, help='Output colored masks/overlays directory')
    return parser.parse_args()

args = parse_args()

DATA_DIR = args.data_dir
MASKS_DIR = os.path.join(DATA_DIR, 'layer_masks')
VIDEOS_DIR = os.path.join(DATA_DIR, 'layer_videos')
OUTPUT_IMAGES = args.output_images or os.path.join(DATA_DIR, 'segmentation_images')
OUTPUT_MASKS = args.output_masks or os.path.join(DATA_DIR, 'segmentation_masks')
OUTPUT_MASKS_COLOR = args.output_masks_color or os.path.join(DATA_DIR, 'segmentation_masks_color')

os.makedirs(OUTPUT_IMAGES, exist_ok=True)
os.makedirs(OUTPUT_MASKS, exist_ok=True)
os.makedirs(OUTPUT_MASKS_COLOR, exist_ok=True)

def get_label_colormap(num_labels):
    cmap = (plt.colormaps['tab20'](np.arange(num_labels))[:, :3] * 255).astype(np.uint8)
    cmap[0] = [0,0,0]
    return cmap

object_name_to_global_id = OrderedDict()
for subdir in sorted(os.listdir(MASKS_DIR)):
    names_json = os.path.join(MASKS_DIR, subdir, 'names.json')
    if os.path.exists(names_json):
        with open(names_json, 'r') as f:
            names_map = json.load(f)
        for obj_name in names_map.values():
            if obj_name not in object_name_to_global_id:
                object_name_to_global_id[obj_name] = len(object_name_to_global_id) + 1

# flip for alternate lookup
global_id_to_obj_name = OrderedDict({value:key for key, value in object_name_to_global_id.items()})
colormap = get_label_colormap(len(object_name_to_global_id) + 1)

for subdir in sorted(os.listdir(MASKS_DIR)):
    masks_json_path = os.path.join(MASKS_DIR, subdir, 'masks.json')
    names_json_path = os.path.join(MASKS_DIR, subdir, 'names.json')
    video_file = os.path.join(VIDEOS_DIR, f'{subdir}.mp4')
    if not all(map(os.path.exists, [masks_json_path, names_json_path, video_file])):
        print(f"Skipping {subdir}, missing files.")
        continue

    with open(names_json_path, 'r') as f:
        local_names = json.load(f)

    local_to_global = {int(k): object_name_to_global_id[v] for k,v in local_names.items()}
    global_to_local = {v:k for k,v in local_to_global.items()}

    with open(masks_json_path, 'r') as f:
        masks_data = json.load(f)

    mask_frame_indices = sorted([int(k) for k in masks_data.keys()])
    decoded_masks = []
    for frame_idx in mask_frame_indices:
        mask_info = masks_data[str(frame_idx)]
        height, width = mask_info['results'][0]['mask']['size']
        mask_img = np.zeros((height, width), dtype=np.uint8)

        # This is important! The mask application order should be 1) weld head, 2) background, etc... such that the large items don't overwrite the small details.
        for global_obj in object_name_to_global_id.items():
            global_name, global_id = global_obj
            if global_id in global_to_local:
                local_id = global_to_local[global_id]
                obj = mask_info['results'][local_id]
                rle = {"counts": obj['mask']['counts'].encode('utf-8'), "size": obj['mask']['size']}
                mask_img[mask_utils.decode(rle) == 1] = global_id

        decoded_masks.append(mask_img)

    cap = cv2.VideoCapture(video_file)
    num_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Mapping masks to video frames using linspace
    video_indices = np.linspace(0, num_video_frames - 1, num=len(mask_frame_indices), dtype=int)

    NUM_SAMPLES = min(10, num_video_frames)
    sample_indices_set = set(np.linspace(0, num_video_frames - 1, NUM_SAMPLES, dtype=int))

    for mask_idx, vframe_idx in tqdm(zip(range(len(mask_frame_indices)), video_indices), total=len(video_indices), desc=f'Processing {subdir}'):
        interp_mask = decoded_masks[mask_idx]

        mask_out_path = os.path.join(OUTPUT_MASKS, f'seg_{subdir}_{vframe_idx:05d}.png')
        cv2.imwrite(mask_out_path, interp_mask.astype(np.uint8))

        cap.set(cv2.CAP_PROP_POS_FRAMES, vframe_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        img_out_path = os.path.join(OUTPUT_IMAGES, f'rgb_{subdir}_{vframe_idx:05d}.png')
        cv2.imwrite(img_out_path, frame)

        if vframe_idx in sample_indices_set:
            color_mask_bgr = colormap[interp_mask][..., ::-1]

            if color_mask_bgr.shape[:2] != frame.shape[:2]:
                color_mask_bgr = cv2.resize(color_mask_bgr, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

            overlay = cv2.addWeighted(frame, 0.6, color_mask_bgr, 0.4, 0)
            color_path = os.path.join(OUTPUT_MASKS_COLOR, f'{subdir}_{vframe_idx:05d}_maskcolor.png')
            overlay_path = os.path.join(OUTPUT_MASKS_COLOR, f'{subdir}_{vframe_idx:05d}_overlay.png')
            cv2.imwrite(color_path, color_mask_bgr)
            cv2.imwrite(overlay_path, overlay)

    cap.release()
