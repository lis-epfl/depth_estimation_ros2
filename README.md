# depth_estimation_ros2

A ROS 2 package for real-time stereo depth estimation using ONNX models (S2M2, Hitnet, CREStereo).
This node processes a 4-camera concatenated stream into 3D point clouds using GPU acceleration (TensorRT/CUDA).

## Features
* **Multi-Model Support:** Switch between S2M2, Hitnet, and CREStereo architectures.
* **Parallel Processing:** Option to run inference on multiple stereo pairs concurrently using NVIDIA Multi-Stream or sequentially.
* **Efficient Pipeline:** Handles image splitting, rectification, inference, and point cloud generation.
* **Headless:** Publishes compressed debug imagery to ROS topics instead of GUI windows.

## Prerequisites
* **ROS 2 Humble** (or compatible)
* **CUDA & TensorRT** (Required for GPU acceleration)
* **Dependencies:**
  - `libopencv-dev`
  - `libpcl-dev`
  - `libyaml-cpp-dev`
  - `onnxruntime` (GPU version recommended)
* **External Packages:**
  - [`oak_ffc_4p_driver_ros2`](https://github.com/lis-epfl/oak_ffc_4p_driver_ros2)
  - [`quadcam_calibration_tool`](https://github.com/lis-epfl/quadcam_calibration_tool/)

## Installation & Setup

### 1. Clone the repository
```bash
cd ~/ros2_ws/src
git clone https://github.com/lis-epfl/depth_estimation_ros2 depth_estimation_ros2
```

### 2. Download Models
```bash
python3 `src/depth_estimation_ros2/scripts/download_models.py`
```

### 3. Setup Calibration Maps
Copy generated calibration maps into:

```
`src/depth_estimation_ros2/config/final_maps_<RES>/`
```

### 4. Build
```bash
cd ~/ros2_ws
colcon build --symlink-install --packages-select depth_estimation_ros2
source install/setup.bash
```

## Configuration

### Camera Driver Configuration
```yaml
compress_images: false
```

### Depth Node Configuration (`config/depth_params.yaml`)

#### Critical Paths

| Parameter | Description |
|----------|-------------|
| `calibration_base_path` | Absolute path to calibration folder |
| `calibration_resolution` | Resolution suffix (e.g., `224_224`) |
| `onnx_model_path` | Path to ONNX model |
| `transform_config_path` | Path to `cam0_to_centroid.yaml` |

#### Runtime Tuning

| Parameter | Default | Description |
|----------|---------|-------------|
| `run_parallel` | true | Parallel ONNX inference |
| `model_type` | s2m2 | hitnet / crestereo / s2m2 |
| `execution_provider` | tensorrt | cuda / cpu |
| `verbose` | true | Publish debug grid |
| `filter_disparity` | true | S2M2 filtering |
| `publish_combined_pointcloud` | false | Merge clouds |

## Usage
```bash
ros2 launch depth_estimation_ros2 depth_estimation.launch.py
```

## Topics

| Type | Topic | Description |
|------|--------|-------------|
| Sub | `/oak_ffc_4p_driver_node/image_raw` | Concatenated 4-camera image |
| Pub | `/depth/pointcloud/pair_X_Y` | Stereo pair clouds |
| Pub | `/depth/pointcloud/combined` | Full cloud |
| Pub | `/depth/debug_grid/compressed` | Debug disparity grid |
