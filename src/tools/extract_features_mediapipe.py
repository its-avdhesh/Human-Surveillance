#!/usr/bin/env python3
"""
Extract skeletons from videos in `../test_data/` using MediaPipe,
convert to the project's skeleton format, generate time-serial features
using existing `FeatureGenerator`, and save features plus a metadata CSV
for labeling.

Outputs:
 - ../datasets/realtime_action_recognition/extracted_data/features_X.csv
 - ../datasets/realtime_action_recognition/extracted_data/features_meta.csv

Run from `src/` directory.
"""
import os
import glob
import csv
import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception:
    print('mediapipe not installed. Run: pip install mediapipe')
    raise

from classifier.dnn.feature_procs import FeatureGenerator
from utils.config import Config


def mp_to_body_joints(landmarks, w, h):
    """Map MediaPipe landmarks to 13 body joints expected by feature procs.
    Returns flattened array of 13 joints (x,y) normalized to [0,1]. Missing
    joints are set to 0.
    Order: neck, r_shoulder, r_elbow, r_wrist, l_shoulder, l_elbow, l_wrist,
           r_hip, r_knee, r_ankle, l_hip, l_knee, l_ankle
    """
    # indices in mediapipe pose
    idx = {}
    # left/right shoulder/elbow/wrist
    idx['l_shoulder'] = 11
    idx['r_shoulder'] = 12
    idx['l_elbow'] = 13
    idx['r_elbow'] = 14
    idx['l_wrist'] = 15
    idx['r_wrist'] = 16
    idx['l_hip'] = 23
    idx['r_hip'] = 24
    idx['l_knee'] = 25
    idx['r_knee'] = 26
    idx['l_ankle'] = 27
    idx['r_ankle'] = 28

    def get_lm(key):
        i = idx[key]
        if i >= len(landmarks):
            return 0.0, 0.0
        lm = landmarks[i]
        if getattr(lm, 'visibility', None) is not None and lm.visibility < 0.2:
            return 0.0, 0.0
        return lm.x, lm.y

    # compute neck as midpoint between shoulders
    lx, ly = get_lm('l_shoulder')
    rx, ry = get_lm('r_shoulder')
    neck_x = (lx + rx) / 2.0
    neck_y = (ly + ry) / 2.0

    order = [
        ('neck', (neck_x, neck_y)),
        ('r_shoulder', get_lm('r_shoulder')),
        ('r_elbow', get_lm('r_elbow')),
        ('r_wrist', get_lm('r_wrist')),
        ('l_shoulder', get_lm('l_shoulder')),
        ('l_elbow', get_lm('l_elbow')),
        ('l_wrist', get_lm('l_wrist')),
        ('r_hip', get_lm('r_hip')),
        ('r_knee', get_lm('r_knee')),
        ('r_ankle', get_lm('r_ankle')),
        ('l_hip', get_lm('l_hip')),
        ('l_knee', get_lm('l_knee')),
        ('l_ankle', get_lm('l_ankle')),
    ]

    flat = []
    for _, (x, y) in order:
        # landmarks are normalized by mediapipe already (0-1)
        flat.append(float(x))
        flat.append(float(y))
    return np.array(flat, dtype=float)


def main():
    cfg = Config(config_file='../configs/train_action_recogn_pipeline.yaml')
    window_size = cfg.window_size

    src_videos = sorted(glob.glob(os.path.join('..', 'test_data', '*.mp4')))
    if len(src_videos) == 0:
        print('No videos found in ../test_data/')
        return

    # output folder
    out_dir = os.path.abspath(os.path.join('..', 'datasets', 'realtime_action_recognition', 'extracted_data'))
    os.makedirs(out_dir, exist_ok=True)

    mp_pose = mp.solutions.pose

    all_X0 = []
    all_Y0 = []
    video_indices = []
    meta = []  # list of (video_file, frame_idx)

    vid_id = 0
    for vid in src_videos:
        print('Processing', vid)
        cap = cv2.VideoCapture(vid)
        with mp_pose.Pose(static_image_mode=False, model_complexity=1) as pose:
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                h, w = frame.shape[:2]
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)
                if results.pose_landmarks:
                    sk = mp_to_body_joints(results.pose_landmarks.landmark, w, h)
                else:
                    sk = np.zeros((13*2,), dtype=float)
                # Build a skeleton-like 36-length vector where body joints placed at indices 2:2+26
                arr36 = np.zeros((36,), dtype=float)
                arr36[2:2+13*2] = sk
                all_X0.append(arr36)
                all_Y0.append(-1)  # placeholder
                video_indices.append(vid_id)
                meta.append((os.path.basename(vid), frame_idx))
                frame_idx += 1
        cap.release()
        vid_id += 1

    X0 = np.array(all_X0)
    Y0 = np.array(all_Y0)
    video_indices = np.array(video_indices)
    print('Total frames processed:', len(X0))

    # Extract time-serial features using FeatureGenerator logic while recording mapping
    FG = None
    features = []
    features_meta = []
    for i in range(len(video_indices)):
        if i == 0 or video_indices[i] != video_indices[i-1]:
            FG = FeatureGenerator(window_size)
        success, feat = FG.add_cur_skeleton(X0[i])
        if success:
            features.append(feat)
            # map this feature to the originating video/frame (we associate with current frame)
            features_meta.append((len(features)-1, meta[i][0], meta[i][1]))

    features = np.array(features)
    print('Extracted features shape:', features.shape)

    # Save features and meta for labeling
    features_X_path = os.path.join(out_dir, 'features_X.csv')
    np.savetxt(features_X_path, features, fmt='%.5f')
    print('Saved features to', features_X_path)

    meta_path = os.path.join(out_dir, 'features_meta.csv')
    with open(meta_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['feature_idx', 'video_file', 'frame_idx', 'label'])
        for idx, vf, fr in features_meta:
            writer.writerow([idx, vf, fr, ''])
    print('Saved features metadata to', meta_path)


if __name__ == '__main__':
    main()
