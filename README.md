# depth_estimation_ros2

A ROS 2 package for real-time stereo depth estimation using ONNX models (S2M2, Hitnet, CREStereo).
This node processes a 4-camera concatenated stream into 3D point clouds using GPU acceleration (TensorRT/CUDA).

![Calibration Visualization](imgs/disparity.gif)

## Features
* **Multi-Model Support:** Switch between S2M2, Hitnet, and CREStereo architectures.
* **Parallel Processing:** Option to run inference on multiple stereo pairs concurrently using NVIDIA Multi-Stream or sequentially.
* **Efficient Pipeline:** Handles image splitting, rectification, inference, and point cloud generation.
* **Headless:** Publishes compressed debug imagery to ROS topics instead of GUI windows.
* **Per-Drone Depth Corrections:** Support for calibration-specific baseline and offset corrections.

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
Copy generated calibration maps into: `src/depth_estimation_ros2/config/final_maps_<RES>/` e.g. `final_maps_224_224`.

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

## Depth Correction Calibration

Due to small errors in stereo calibration (baseline estimation, principal point alignment, rectification), the estimated depth may have systematic errors. This package supports per-drone corrections via a `depth_corrections.yaml` file in each calibration folder.

### Mathematical Background

The stereo depth equation is:
```
Z = (f × b) / d
```
Where `Z` is depth, `f` is focal length, `b` is baseline, and `d` is disparity.

Two types of systematic errors can occur:
1. **Baseline/focal length error**: The effective `f×b` differs from calibration
2. **Disparity offset error**: Systematic bias in disparity measurement

These errors combine to give a **linear relationship in inverse-depth space**:
```
1/Z_est = baseline_scale × (1/Z_real) + intercept
```

### Correction Parameters

| Parameter | Meaning | How to obtain |
|-----------|---------|---------------|
| `baseline_scale` | Slope of linear fit in inverse-depth space | Directly from linear regression |
| `offset_factor` | Negative intercept of linear fit | `offset_factor = -intercept` |

**Correction formula:**
```
Z_corrected = baseline_scale × Z_est / (1 + offset_factor × Z_est)
```

**Key advantage:** Both parameters come directly from the linear fit — no need to know focal length or baseline separately!

### Diagnosing Error Type

Calculate the ratio `r = Z_real / Z_est` at multiple distances:

| Ratio Pattern | Meaning | Correction Needed |
|---------------|---------|-------------------|
| Constant ratio at all distances | Pure baseline error | `baseline_scale` only (`offset_factor ≈ 0`) |
| Ratio increases with distance | Underestimates at far range | Both parameters, `offset_factor > 0` |
| Ratio decreases with distance | Overestimates at far range | Both parameters, `offset_factor < 0` |

### Measurement Procedure

#### Equipment Needed
- Motion capture system (Vicon, OptiTrack, etc.) OR accurate tape measure
- Flat wall or target at known distances
- Access to Foxglove (recommended) or RViz2 for pointcloud visualization

#### Step-by-Step Calibration

1. **Setup the environment**
   ```bash
   # Terminal 1: Start the camera driver
   ros2 launch oak_ffc_4p_driver_ros2 oak_ffc_4p_driver.launch.py

   # Terminal 2: Start depth estimation (with verbose mode)
   ros2 launch depth_estimation_ros2 depth_estimation.launch.py

   # Terminal 3: Open RViz2
   rviz2
   # Or if using Foxglove (use the corresponding ros domain id and open foxglove app):
   ROS_DOMAIN_ID=0 ros2 launch foxglove_bridge foxglove_bridge_launch.xml
   ```

2. **Configure RViz2**
   - Add PointCloud2 displays for each pair: `/depth/pointcloud/pair_0_1`, etc.
   - Set the fixed frame to match your `pointcloud_frame_id`
   - Use different colors for each pair to distinguish them

   If using Foxglove:
   - Add 3D panel.
   - In `Frame` set the display frame to match your `pointcloud_frame_id`.
   - In `Topics` you can choose the pointclouds to visualize and set different colors.

3. **Collect measurements at multiple distances**

   Position a flat surface (wall) at known distances from the camera. **Minimum 2 distances required** (more recommended for verification):

   | Distance | Purpose |
   |----------|---------|
   | ~1.5m | Near range reference |
   | ~2.5m | Mid range |
   | ~3.5m | Mid-far range |
   | ~4.5m+ | Far range (most sensitive to offset errors) |

4. **Record ground truth and estimated depths**

   For each stereo pair and each distance:
   - **Ground truth (Z_real)**: Use mocap position or tape measure
   - **Estimated (Z_est)**: Read from RViz2 (hover over pointcloud or use "Publish Point" tool). In Foxglove click on the ruler logo on the top right of the 3D panel to measure the exact distance.

   Example data format:
   ```
   Z_real (m) | Pair 0_1 | Pair 1_2 | Pair 2_3 | Pair 3_0
   -----------|----------|----------|----------|----------
   1.85       | 1.82     | 1.70     | 1.90     | 1.80
   3.70       | 3.68     | 3.07     | 4.11     | 3.75
   ```

5. **Run the calibration script**

   Use the provided calibration script:
   ```bash
   python3 scripts/depth_calibration.py
   ```

   The script will:
   - Prompt for your measurements
   - Fit the linear model in inverse-depth space
   - Output `baseline_scale` and `offset_factor` for each pair
   - Verify the correction produces accurate depths
   - Generate the YAML configuration

6. **Create the corrections file**

   Save the output as `depth_corrections.yaml` in your calibration folder:
   ```yaml
   # config/final_maps_256_160/depth_corrections.yaml
   # Correction: Z_corrected = baseline_scale * Z_est / (1 + offset_factor * Z_est)

   depth_correction:
     # Order matches stereo_pairs: ["0_1", "1_2", "2_3", "3_0"]
     baseline_scales: [1.027532, 0.971259, 1.047125, 1.068889]
     offset_factors: [0.005972, -0.063231, 0.039698, 0.022222]
   ```

7. **Verify corrections**

   Restart the depth estimation node and repeat measurements to verify accuracy.

### Correction Parameter Reference

| Scenario | `baseline_scale` | `offset_factor` |
|----------|------------------|-----------------|
| Depths too short (underestimate) | > 1.0 | > 0 (if error grows with distance) |
| Depths too long (overestimate) | < 1.0 | < 0 (if error grows with distance) |
| No correction needed | 1.0 | 0.0 |

### File Location

```
config/
└── final_maps_256_160/
    ├── final_map_config_0_1.yaml
    ├── final_map_config_1_2.yaml
    ├── final_map_config_2_3.yaml
    ├── final_map_config_3_0.yaml
    ├── final_rectified_to_fisheye_map_*.yml
    └── depth_corrections.yaml          ← Per-drone corrections
```

If `depth_corrections.yaml` is missing, the node uses default values (no correction) and logs a warning.

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

## Troubleshooting

### Depth corrections not loading
- Check that `depth_corrections.yaml` is in the correct calibration folder
- Verify the YAML syntax (arrays must have exactly 4 elements)
- Check node logs for "Loading depth corrections from:" message

### Large depth errors on one pair only
- That pair likely has calibration issues
- Consider re-running stereo calibration for that pair
- Use larger correction values for that specific pair

### Correction doesn't improve accuracy
- Verify you have at least 2 distance measurements
- Check that the linear model assumption holds (errors should follow pattern described above)
- Try adding more measurement points to verify fit quality
