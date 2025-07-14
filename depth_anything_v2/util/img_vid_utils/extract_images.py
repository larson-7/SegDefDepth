import os
import cv2
import random
from video_cropping import detect_crop_params, crop_video_ffmpeg
from pathlib import Path

data_dir = "training_data/layer_videos"
data_dir = os.path.join(os.getcwd(), data_dir)
assert os.path.exists(data_dir), "data directory does not exist, create data directory in this repository and add videos"

croppped_dir = "cropped"
cropped_dir = os.path.join(data_dir, croppped_dir)
# create cropped dir if it doesn't exist
os.makedirs(croppped_dir, exist_ok=True)

frame_dir = "frames"
frame_dir = os.path.join(data_dir, frame_dir)
# create cropped dir if it doesn't exist
os.makedirs(frame_dir, exist_ok=True)

# get list of videos in top level data directory
videos = [item for item in os.listdir(data_dir) if item.endswith(".mp4") and os.path.isfile(os.path.join(data_dir, item))]
print(f"Found {len(videos)} videos")

for video in videos:
    print(f"Processing {video}")

    # 1) crop video
    video_path = os.path.join(data_dir, video)
    cropped_video_path = os.path.join(croppped_dir, video)
    # detect crop parameters
    crop = detect_crop_params(video_path)
    # apply cropping
    crop_video_ffmpeg(video_path, cropped_video_path, crop)

    # 2) open cropped video and extract frames
    # open input video
    cap = cv2.VideoCapture(cropped_video_path)
    if not cap.isOpened():
        print("Error: Cannot open input video.")
        exit()

    # get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Total frames in {video}:{total_frames}")

    frame_idx = 0
    stride = 5
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # keep every 5th frame
        if frame_idx % stride == 0:
            out_dir = os.path.join(frame_dir, video.replace(".mp4", ""))
            os.makedirs(out_dir, exist_ok=True)
            cv2.imwrite(os.path.join(out_dir, f"{frame_idx:06d}.jpg"), frame)
        frame_idx += 1


