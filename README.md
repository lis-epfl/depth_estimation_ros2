# depth_estimation_ros2

A ROS 2 package for real-time stereo depth estimation using ONNX models (S2M2, Hitnet, CREStereo).
This node processes a 4-camera concatenated stream into 3D point clouds using GPU acceleration (TensorRT/CUDA).

![Calibration Visualization](imgs/disparity.gif)

## Features
* **Multi-Model Support:** Switch between S2M2, Hitnet, and CREStereo architectures.
* **Parallel Processing:** Option to run inference on multiple stereo pairs concurrently using NVIDIA Multi-Stream or sequentially.
* **Efficient Pipeline:** Handles image splitting, rectification, inference, and point cloud generation.
* **Headless:** Publishes compressed debug imagery to ROS topics instead of GUI windows.
* **Per-Drone Depth Corrections:** Support for calibration-specific baseline and disparity corrections.

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

### Understanding the Corrections

The stereo depth equation is:
```
Z = (f × B) / d
```
Where `Z` is depth, `f` is focal length, `B` is baseline, and `d` is disparity.

Two correction parameters are available:

| Parameter | Effect | When to Use |
|-----------|--------|-------------|
| `baseline_scale` | Multiplies the baseline: `Z_corrected = Z_original × scale` | Depth is consistently over/underestimated by a fixed percentage |
| `disparity_offset` | Subtracts from disparity: `d_corrected = d - offset` | Error increases with distance (far objects have larger errors) |

### Symptom Diagnosis

| Observation | Likely Cause | Fix |
|-------------|--------------|-----|
| All depths too short (underestimate) | Baseline too small | `baseline_scale > 1.0` |
| All depths too long (overestimate) | Baseline too large | `baseline_scale < 1.0` |
| Near objects accurate, far objects wrong | Disparity offset | Adjust `disparity_offset` |
| Error ratio (real/est) increases with distance | Combined error | Need both corrections |

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
   # Or if using Foxglove (use the correspoding ros domain id and open foxglove app):
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

   Position a flat surface (wall) at known distances from the camera. For each distance:

   | Distance | Why |
   |----------|-----|
   | ~1.2m | Near range reference |
   | ~2.5m | Mid range |
   | ~3.5m | Mid-far range |
   | ~4.5m+ | Far range (most sensitive to disparity offset) |

4. **Record ground truth and estimated depths**

   For each stereo pair and each distance:
   - **Ground truth**: Use mocap position or tape measure
   - **Estimated**: Read from RViz2 (hover over pointcloud or use "Publish Point" tool). In Foxglove click on the ruler logo on the top right of the 3D panel to measure the exact distant.

   Example data format:
   ```
   Distance | Real (m) | Pair 0_1 | Pair 1_2 | Pair 2_3 | Pair 3_0
   ---------|----------|----------|----------|----------|----------
   Near     | 1.20     | 1.04     | 0.90     | 1.18     | 1.05
   Mid      | 2.45     | 2.00     | 1.60     | 2.40     | 2.00
   Mid-far  | 3.61     | 2.76     | 2.06     | 3.61     | 2.79
   Far      | 4.60     | 3.27     | 2.33     | 4.60     | 3.54
   ```

5. **Analyze the error pattern**

   Calculate the ratio `Real / Estimated` for each measurement:

   ```
   Distance | Pair 0_1 Ratio | Pair 1_2 Ratio | ...
   ---------|----------------|----------------|----
   Near     | 1.15           | 1.33           |
   Mid      | 1.23           | 1.53           |
   Mid-far  | 1.31           | 1.75           |
   Far      | 1.41           | 1.97           |
   ```

   - **Constant ratio** → Only `baseline_scale` needed (= average ratio)
   - **Increasing ratio with distance** → Both corrections needed

6. **Calculate correction parameters**

   **Simple case (constant ratio):**
   ```
   baseline_scale = average(Real / Estimated)
   disparity_offset = 0.0
   ```

   **Complex case (ratio increases with distance):**

   The error model in inverse-depth space is linear:
   ```
   1/Z_est = (1/baseline_scale) × (1/Z_real) + offset_factor
   ```

   Fit this linear model to your data points `(1/Z_real, 1/Z_est)` to get:
   - `baseline_scale` from the slope
   - `disparity_offset` from the intercept (requires knowing focal length)

   **Practical approximation:**
   - Start with `baseline_scale = ratio at near distance`
   - Adjust `disparity_offset` until far distance is correct
   - Iterate if needed

7. **Create the corrections file**

   Create `depth_corrections.yaml` in your calibration folder:
   ```yaml
   # config/final_maps_256_160/depth_corrections.yaml
   depth_correction:
     # Order matches stereo_pairs: ["0_1", "1_2", "2_3", "3_0"]
     baseline_scales: [1.07, 1.45, 0.92, 1.04]
     disparity_offsets: [1.0, 0.5, -0.9, 1.0]
   ```

8. **Verify corrections**

   Restart the depth estimation node and repeat measurements to verify accuracy.

### Correction Parameter Reference

| Scenario | `baseline_scale` | `disparity_offset` |
|----------|----------------|------------------|
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
