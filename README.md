![til](figures\demo_reconstruction_3d.gif)
# MultiCam3D-Pose

> Multi-camera 3D human pose estimation and skeleton reconstruction system

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![YOLO](https://img.shields.io/badge/YOLO-v26-red.svg)](https://github.com/ultralytics/ultralytics)
[![Status](https://img.shields.io/badge/Status-In%20Development-yellow.svg)]()

## Overview

**MultiCam3D-Pose** is a computer vision system that reconstructs 3D human skeletons from multiple synchronized camera views. The system uses YOLO-based 2D pose detection, epipolar geometry for skeleton matching across views, and triangulation for 3D reconstruction.

### Key Features

- **Multi-camera synchronization** - Processes frames from multiple calibrated cameras
- **2D pose detection** - Uses YOLOv26-pose for robust keypoint detection
- **Cross-view matching** - Matches skeletons across cameras using epipolar geometry
- **3D reconstruction** - Triangulates 2D keypoints into 3D coordinates via SVD
- **Real-time visualization** - Live 3D skeleton visualization with camera frustums
- **Distortion correction** - Automatic camera undistortion using calibration data

##  Project Status

**Current Stage:** Development 

### Implemented
- Multi-camera video/image processing
- 2D skeleton detection with YOLO
- Fundamental matrix calculation
- Epipolar constraint-based matching
- 3D reconstruction via triangulation
- Real-time 3D visualization

### In Progress
- **Improved matching algorithm** - Refining cross-view correspondence
- **Temporal tracking** - Assigning unique IDs across frames
- **ID persistence** - Maintaining skeleton identity over time


## Architecture

```
┌─────────────────┐
│  Video Input    │  ← Multiple camera feeds
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ VideoProcessor  │  ← Undistortion
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ SkeletonDetector│  ← YOLO 2D pose detection
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ SkeletonMatcher │  ← Epipolar matching
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Reconstructor3D │  ← Triangulation
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Visualizer    │  ← 3D rendering
└─────────────────┘
```

## Technical Details

### Matching Pipeline

The system employs a three-stage filtering approach:

1. **Epipolar Filtering** - Validates skeleton correspondence using fundamental matrices
   - Maximum epipolar distance: 15 pixels
   - Minimum matching joints: 5 keypoints

2. **3D Distance Filtering** - Computes intersection points and validates spatial proximity
   - Maximum 3D intersection distance: 7 pixels
   - Weighted scoring: 75% distance, 25% confidence

3. **Completeness Filtering** - Ensures sufficient keypoints for reliable reconstruction
   - Minimum keypoints required: 8

### 3D Reconstruction

- **Method:** Direct Linear Transform
- **Input:** 2D keypoints from ≥2 cameras
- **Output:** Homogeneous 3D coordinates
- **Coordinate system:** World coordinates centered on calibration origin

## Installation

### Prerequisites

```bash
Python >= 3.12 
```

### Dependencies

```bash
pip install -r requirements.txt   
```

### YOLO Model

Download the YOLOv26-pose model:
```bash
# Place the model in ./models/
# Supported formats: .pt (PyTorch) or .engine (TensorRT)
```

## Project Structure

```
MultiCam3D-Pose/
├── skeleton_tracker_main.py    # Main entry point
├── video_processor.py           # Multi-camera frame handling
├── skeletons.py                 # YOLO pose detection
├── skeleton_matcher.py          # Cross-view matching
├── reconstructor_3d.py          # 3D triangulation
├── fundamental_matrices.py      # Epipolar geometry
├── visualizer.py                # 3D rendering
├── utils.py                     # Helper functions
├── calib_cameras/               # Camera calibration files
│   ├── calib_rt1.npz
│   ├── calib_rt2.npz
│   └── ...
├── models/                      # YOLO models
│   └── yolo26m-pose.engine
└── videos/                      # Input videos
    ├── camera_1.avi
    └── ...
```

## Configuration

Edit `config` dictionary in `skeleton_tracker_main.py`:

```python
config = {
    "num_cameras": 4,           # Number of camera views
    "n_keypoints": 18,          # YOLO keypoint count (COCO format)
    
    # Matching thresholds
    "max_epipolar_dist": 15,    # Pixels
    "min_matching_joints": 5,   # Keypoints
    "max_intersection_dist": 7, # Pixels

    "weight_distance": 0.75,
    "weight_score": 0.25,
      
    "min_keypoints_for_grouping": 8, # Keypoints
    
    # Paths
    "calib_path": "./calib_cameras",
    "video_path": "./videos",
    "yolo_model": "./models/yolo26m-pose.engine"
}
```

---