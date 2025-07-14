import os
import shutil
import random

ROOT = '/Users/sheanscottjr/omscs/cs8903/SegDefDepth/training_data'
IMG_DIR = os.path.join(ROOT, 'segmentation_images')
MASK_DIR = os.path.join(ROOT, 'segmentation_masks')
SPLIT_DIR = os.path.join(ROOT, 'official_splits')

# Clean out old splits if needed
for split in ['train', 'test']:
    split_dir = os.path.join(SPLIT_DIR, split)
    if os.path.exists(split_dir):
        shutil.rmtree(split_dir)

# Step 1: Group frame ids by video prefix using your parsing
all_basenames = [f for f in os.listdir(IMG_DIR) if f.endswith('.png') and '_' in f]
video_to_frameids = {}
pairs = []
for fname in all_basenames:
    type_, video, frameid = fname.split('_', 2)
    frameid = frameid.replace(".png", "")
    video_to_frameids.setdefault(video, []).append(frameid)
    pairs.append((video, frameid))

# Step 2: Shuffle and IID split at frame level
random.seed(42)
random.shuffle(pairs)
split_idx = int(0.8 * len(pairs))
train_pairs = pairs[:split_idx]
test_pairs = pairs[split_idx:]

# Step 3: Copy files
def copy_pairs(pairs, split):
    for video, frameid in pairs:
        split_video_dir = os.path.join(SPLIT_DIR, split, video)
        os.makedirs(split_video_dir, exist_ok=True)
        img_src = os.path.join(IMG_DIR, f'rgb_{video}_{frameid}.png')
        mask_src = os.path.join(MASK_DIR, f'seg_{video}_{frameid}.png')
        img_dst = os.path.join(split_video_dir, f'rgb_{video}_{frameid}.png')
        mask_dst = os.path.join(split_video_dir, f'seg_{video}_{frameid}.png')
        if os.path.exists(img_src):
            shutil.copy(img_src, img_dst)
        if os.path.exists(mask_src):
            shutil.copy(mask_src, mask_dst)

copy_pairs(train_pairs, 'train')
copy_pairs(test_pairs, 'test')

print("IID split done! Each train/test/[video]/ folder contains rgb_... and seg_... files (frame-level IID split).")
