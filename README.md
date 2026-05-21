# Human Surveillance - Action Recognition System

A comprehensive real-time human action recognition system that combines pose estimation, person tracking, and action classification for surveillance applications.

## Overview

This project implements a complete pipeline for human action recognition in video streams, featuring:
- **Pose Estimation**: TRTPose for extracting human skeleton keypoints
- **Person Tracking**: DeepSORT algorithm for multi-person tracking across frames
- **Action Classification**: DNN-based classifier for recognizing human actions
- **Real-time Inference**: Demo application for live video processing

## Features

- Multi-person pose estimation using TRTPose (DenseNet121 backbone)
- Robust person tracking with DeepSORT and ReID models (Market1501/MARS datasets)
- Action recognition for 9 action classes: stand, walk, run, jump, sit, squat, kick, punch, wave
- Configurable pipeline with YAML configuration files
- Support for both video files and webcam input
- Real-time visualization with debugging capabilities
- Training pipeline for custom action recognition models

## Project Structure

```
Human-Surveillance/
├── assets/                 # Demo images and visualizations
├── configs/               # Configuration files
│   ├── infer_trtpose_deepsort_dnn.yaml    # Inference configuration
│   ├── train_action_recogn_pipeline.yaml # Training pipeline configuration
│   └── train_reid.yaml                    # ReID training configuration
├── export_models/        # Exported model files
├── src/                   # Source code
│   ├── demo.py           # Main inference demo
│   ├── train_reid.py     # ReID model training
│   ├── s1_extract_trtpose_skeletons.py   # Step 1: Extract skeletons
│   ├── s2_combine_skeletons_txt.py      # Step 2: Combine skeletons
│   ├── s3_gen_features.py                # Step 3: Generate features
│   ├── s4_train_classifier.py            # Step 4: Train classifier
│   └── lib/              # Library modules
│       ├── classifier/   # Action classification modules
│       ├── pose_estimation/  # Pose estimation modules
│       ├── tracker/      # Tracking modules (DeepSORT)
│       └── utils/        # Utility functions
├── test_data/            # Test data samples
├── weights/              # Pre-trained model weights
│   ├── classifier/       # Action classifier models
│   ├── pose_estimation/  # Pose estimation models
│   └── tracker/          # ReID tracking models
├── requirements.txt      # Python dependencies
└── LICENSE              # MIT License
```

## Installation

### Prerequisites

- Python 3.7+
- CUDA-capable GPU (recommended for real-time performance)
- OpenCV 4.2+

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Human-Surveillance
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download pre-trained weights and place them in the `weights/` directory:
   - Pose estimation model: `weights/pose_estimation/trtpose/densenet121_baseline_att_256x256_B_epoch_160.pth`
   - ReID model: `weights/tracker/deepsort/siamese_mars.trt`
   - Action classifier: `weights/classifier/dnn/action_classifier2.pkl`

## Usage

### Running the Demo

Run the inference demo with a video file:
```bash
cd src
python demo.py --source /path/to/video.mp4 --task action --config ../configs/infer_trtpose_deepsort_dnn.yaml --save_folder ../output
```

Run with webcam (default):
```bash
cd src
python demo.py --task action --config ../configs/infer_trtpose_deepsort_dnn.yaml
```

### Demo Arguments

- `--task`: Inference task - `pose`, `track`, or `action` (default: `action`)
- `--source`: Input video path or webcam index (default: `0` for webcam)
- `--config`: Path to configuration file (default: `../configs/infer_trtpose_deepsort_dnn.yaml`)
- `--save_folder`: Output folder for saving results (default: `../output`)
- `--draw_kp_numbers`: Draw keypoint numbers on visualization (default: `False`)
- `--debug_track`: Visualize tracker debugging information (default: `True`)

### Training Pipeline

To train a custom action recognition model, follow these steps:

1. **Extract Skeletons** (Step 1):
```bash
cd src
python s1_extract_trtpose_skeletons.py --config ../configs/train_action_recogn_pipeline.yaml
```

2. **Combine Skeletons** (Step 2):
```bash
python s2_combine_skeletons_txt.py --config ../configs/train_action_recogn_pipeline.yaml
```

3. **Generate Features** (Step 3):
```bash
python s3_gen_features.py --config ../configs/train_action_recogn_pipeline.yaml
```

4. **Train Classifier** (Step 4):
```bash
python s4_train_classifier.py --config ../configs/train_action_recogn_pipeline.yaml
```

### Training ReID Model

Train a person re-identification model:
```bash
cd src
python train_reid.py --config ../configs/train_reid.yaml --gpu 0
```

## Configuration

### Inference Configuration

Edit `configs/infer_trtpose_deepsort_dnn.yaml` to customize:

- **POSE**: Pose estimation settings (model path, size, keypoint filters)
- **TRACKER**: DeepSORT parameters (max distance, IOU threshold, max age)
- **CLASSIFIER**: Action recognition settings (classes, window size, threshold)

### Training Configuration

Edit `configs/train_action_recogn_pipeline.yaml` to customize:

- **classes**: Action classes for recognition
- **window_size**: Number of adjacent frames for feature extraction
- **data_root**: Path to training dataset
- **extract_path**: Path for extracted features

## Supported Actions

The system supports the following action classes (configurable):
- stand
- walk
- run
- jump
- sit
- squat
- kick
- punch
- wave

## Model Architecture

### Pose Estimation
- **Model**: TRTPose
- **Backbone**: DenseNet121
- **Input Size**: 256x256
- **Output**: 18 keypoints per person

### Person Tracking
- **Algorithm**: DeepSORT
- **ReID Models**: WideResNet or SiameseNet
- **Datasets**: Market1501 or MARS
- **Features**: Appearance + motion cues

### Action Classification
- **Model**: DNN Classifier
- **Input**: Skeleton features from pose estimation
- **Window Size**: 5 frames (configurable)
- **Output**: Action class probabilities

## Performance

- Real-time processing on GPU
- Multi-person tracking and action recognition
- Configurable accuracy/speed trade-offs
- Debug visualization for development

## Dependencies

- pillow==8.2.0
- opencv-contrib-python==4.2.0.32
- scikit-learn==0.21.0
- tabulate
- tqdm
- matplotlib
- numpy
- pyyaml
- rich
- termcolor
- fire
- onnx==1.9.0

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Copyright (c) 2021 ZinMoeHtoo

## Acknowledgments

- TRTPose for pose estimation
- DeepSORT for person tracking
- Market1501 and MARS datasets for ReID training

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this project in your research, please cite:
```bibtex
@software{human_surveillance,
  title={Human Surveillance - Action Recognition System},
  author={ZinMoeHtoo},
  year={2021},
  license={MIT}
}
```
