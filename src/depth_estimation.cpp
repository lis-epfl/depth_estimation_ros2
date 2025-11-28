#include "depth_estimation_ros2/depth_estimation.hpp"
#include "yaml-cpp/yaml.h"
#include <opencv2/highgui.hpp>
#include <pcl/common/transforms.h>
#include <stdexcept>

#include <chrono>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>
#include <algorithm>

namespace depth_estimation {

DepthEstimation::DepthEstimation(const rclcpp::NodeOptions &options)
    : Node("depth_estimation_node", options),
      ort_env_(ORT_LOGGING_LEVEL_WARNING, "depth_estimation") {
  RCLCPP_INFO(this->get_logger(), "Initializing Depth Estimation Node...");

  DeclareRosParameters();
  InitializeRosParameters();

  try {
    InitializeModel();
    InitializeCalibrationData();
    InitializeTransform();
  } catch (const std::exception &e) {
    RCLCPP_FATAL(this->get_logger(), "Initialization failed: %s", e.what());
    rclcpp::shutdown();
    return;
  }

  InitializePublishers();
  InitializeSubscribers();
  InitializeServices();
  is_initialized_ = true;
  RCLCPP_INFO(this->get_logger(),
              "Node initialization complete. Waiting for images...");
}

DepthEstimation::~DepthEstimation() {
  RCLCPP_INFO(this->get_logger(), "Shutting down Depth Estimation Node.");
}

// ---------------------------------------------------------
// HELPER FUNCTIONS
// ---------------------------------------------------------

std::string DepthEstimation::SanitizeString(std::string str) {
  std::replace(str.begin(), str.end(), ' ', '_');
  str.erase(std::remove_if(str.begin(), str.end(),
                           [](char c) { return !std::isalnum(c) && c != '_'; }),
            str.end());
  return str;
}

std::string DepthEstimation::GetSystemArchitecture() {
  struct utsname buffer;
  if (uname(&buffer) != 0) return "unknown_arch";
  return std::string(buffer.machine);
}

std::string DepthEstimation::GetGpuName() {
  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
  if (error_id != cudaSuccess) return "unknown_gpu";
  if (deviceCount == 0) return "no_gpu";

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  std::string name = deviceProp.name;
  return SanitizeString(name);
}

// ---------------------------------------------------------
// ROS PARAMETERS
// ---------------------------------------------------------

void DepthEstimation::DeclareRosParameters() {
  this->declare_parameter<bool>("verbose", false);
  this->declare_parameter<bool>("run_parallel", true); // Default: True (Multi-stream)
  this->declare_parameter<bool>("use_compressed_image", true);
  this->declare_parameter<std::string>("input_image_topic",
                                       "/oak_ffc_4p_driver_node/compressed");

  this->declare_parameter<std::string>("transform_config_path", "");
  this->declare_parameter<std::string>("onnx_model_path", "model.onnx");
  this->declare_parameter<std::string>("model_type", "hitnet");
  this->declare_parameter<std::string>("execution_provider", "tensorrt");
  this->declare_parameter<std::string>("output_pointcloud_topic_prefix",
                                       "/depth/pointcloud");
  this->declare_parameter<std::vector<std::string>>(
      "stereo_pairs", {"0_1", "1_2", "2_3", "3_0"});

  this->declare_parameter<std::string>("calibration_base_path", "");
  this->declare_parameter<std::string>("calibration_resolution", "120_160");

  this->declare_parameter<bool>("publish_combined_pointcloud", false);
  this->declare_parameter<std::string>("combined_pointcloud_topic",
                                       "/depth/pointcloud/combined");
  this->declare_parameter<std::string>("pointcloud_frame_id", "drone_centroid");

  this->declare_parameter<bool>("filter_disparity", false);
}

void DepthEstimation::InitializeRosParameters() {
  this->get_parameter("verbose", verbose_);
  this->get_parameter("run_parallel", run_parallel_);
  this->get_parameter("use_compressed_image", use_compressed_image_);
  this->get_parameter("input_image_topic", input_image_topic_);

  this->get_parameter("transform_config_path", transform_config_path_);
  this->get_parameter("onnx_model_path", onnx_model_path_);
  this->get_parameter("model_type", model_type_);
  this->get_parameter("execution_provider", execution_provider_);
  this->get_parameter("stereo_pairs", stereo_pairs_);

  this->get_parameter("calibration_base_path", calibration_base_path_);
  this->get_parameter("calibration_resolution", calibration_resolution_);

  this->get_parameter("publish_combined_pointcloud",
                      publish_combined_pointcloud_);
  this->get_parameter("combined_pointcloud_topic", combined_pointcloud_topic_);
  this->get_parameter("pointcloud_frame_id", pointcloud_frame_id_);

  this->get_parameter("filter_disparity", filter_disparity_);

  if (verbose_) {
      RCLCPP_INFO(this->get_logger(), "Verbose mode ENABLED: Publishing 2x2 Debug Grid.");
  }

  RCLCPP_INFO(this->get_logger(), "Execution Mode: %s", run_parallel_ ? "PARALLEL (Multi-stream)" : "SEQUENTIAL (Single-stream)");

  if (calibration_base_path_.empty()) {
    throw std::runtime_error(
        "'calibration_base_path' parameter cannot be empty.");
  }

  if (model_type_ != "hitnet" && model_type_ != "crestereo" && model_type_ != "s2m2") {
    throw std::runtime_error("Unsupported model_type: '" + model_type_ +
                             "'. Use 'hitnet', 'crestereo', or 's2m2'.");
  }

  if (model_type_ == "s2m2" && filter_disparity_) {
      RCLCPP_INFO(this->get_logger(), "S2M2 Confidence & Occlusion filtering ENABLED.");
  }
}

void DepthEstimation::InitializeTransform() {
  if (transform_config_path_.empty()) {
    throw std::runtime_error(
        "'transform_config_path' parameter cannot be empty.");
  }

  RCLCPP_INFO(this->get_logger(), "Loading required transform from: %s",
              transform_config_path_.c_str());

  YAML::Node config = YAML::LoadFile(transform_config_path_);
  const auto &data = config["transform"]["data"].as<std::vector<float>>();

  if (data.size() != 16) {
    throw std::runtime_error(
        "Transform data must contain 16 elements for a 4x4 matrix.");
  }

  transform_cam0_to_centroid_ =
      Eigen::Map<const Eigen::Matrix<float, 4, 4, Eigen::RowMajor>>(
          data.data());
}

void DepthEstimation::InitializeModel() {
  RCLCPP_INFO(this->get_logger(), "Loading ONNX model from: %s",
              onnx_model_path_.c_str());
  Ort::AllocatorWithDefaultOptions allocator;

  RCLCPP_INFO(this->get_logger(),
              "Inspecting model structure with a temporary CPU session...");
  Ort::SessionOptions inspect_options;
  Ort::Session inspect_session(ort_env_, onnx_model_path_.c_str(),
                               inspect_options);

  size_t num_input_nodes = inspect_session.GetInputCount();
  if (num_input_nodes == 0) throw std::runtime_error("ONNX model has no input nodes.");

  size_t num_output_nodes = inspect_session.GetOutputCount();
  if (num_output_nodes == 0) throw std::runtime_error("ONNX model has no output nodes.");

  input_node_names_.clear();
  input_node_dims_.clear();
  for (size_t i = 0; i < num_input_nodes; i++) {
    auto input_name_ptr = inspect_session.GetInputNameAllocated(i, allocator);
    input_node_names_.emplace_back(input_name_ptr.get());
    input_node_dims_.push_back(inspect_session.GetInputTypeInfo(i)
                                   .GetTensorTypeAndShapeInfo()
                                   .GetShape());
  }

  output_node_names_.clear();
  for (size_t i = 0; i < num_output_nodes; i++) {
    auto output_name_ptr = inspect_session.GetOutputNameAllocated(i, allocator);
    output_node_names_.emplace_back(output_name_ptr.get());
  }

  if (model_type_ == "hitnet") {
    if (input_node_dims_.size() != 1) throw std::runtime_error("HitNet model must have exactly 1 input tensor.");
    model_full_height_ = input_node_dims_[0][2];
    model_full_width_ = input_node_dims_[0][3];
    RCLCPP_INFO(this->get_logger(), "Detected HitNet input resolution: %dx%d", model_full_width_, model_full_height_);

  } else if (model_type_ == "s2m2") {
    if (input_node_dims_.size() != 2) throw std::runtime_error("S2M2 model must have exactly 2 input tensors.");
    model_full_height_ = input_node_dims_[0][2];
    model_full_width_ = input_node_dims_[0][3];
    RCLCPP_INFO(this->get_logger(), "Detected S2M2 input resolution: %dx%d", model_full_width_, model_full_height_);

  } else if (model_type_ == "crestereo") {
    if (input_node_dims_.size() == 4) {
      RCLCPP_INFO(this->get_logger(), "Detected 4-input Crestereo model (Iterative Refinement).");
      model_half_height_ = input_node_dims_[0][2];
      model_half_width_ = input_node_dims_[0][3];
      model_full_height_ = input_node_dims_[2][2];
      model_full_width_ = input_node_dims_[2][3];
    } else if (input_node_dims_.size() == 2) {
      RCLCPP_INFO(this->get_logger(), "Detected 2-input Crestereo model (Single-Pass).");
      model_full_height_ = input_node_dims_[0][2];
      model_full_width_ = input_node_dims_[0][3];
    } else {
      throw std::runtime_error("Unsupported Crestereo model. Must have 2 or 4 input tensors.");
    }
  }

  session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  RCLCPP_INFO(this->get_logger(), "Creating final inference options with '%s' provider...", execution_provider_.c_str());

  if (execution_provider_ == "tensorrt") {
    OrtTensorRTProviderOptions trt_options{};
    trt_options.device_id = 0;
    trt_options.trt_fp16_enable = 1;
    trt_options.trt_engine_cache_enable = 1;

    // --- CACHE DIRECTORY LOGIC ---
    std::string arch = GetSystemArchitecture();
    std::string gpu_name = GetGpuName();
    std::filesystem::path onnx_path(onnx_model_path_);
    std::filesystem::path model_dir = onnx_path.parent_path();
    std::string stem = onnx_path.stem().string();
    std::string cache_dir_name = stem + "_" + gpu_name + "_" + arch + "_trt_engine";
    std::filesystem::path cache_path = model_dir / cache_dir_name;

    RCLCPP_INFO(this->get_logger(), "TRT Cache Target: %s", cache_path.string().c_str());

    if (!std::filesystem::exists(cache_path)) {
        RCLCPP_INFO(this->get_logger(), "Cache directory does not exist. Creating it...");
        try { std::filesystem::create_directories(cache_path); }
        catch (const std::filesystem::filesystem_error& e) {
            RCLCPP_ERROR(this->get_logger(), "Failed to create cache dir: %s. Fallback to /tmp", e.what());
            cache_path = "/tmp/trt_cache";
        }
    }
    std::string cache_path_str = cache_path.string();
    trt_options.trt_engine_cache_path = cache_path_str.c_str();

    try {
      session_options_.AppendExecutionProvider_TensorRT(trt_options);
      RCLCPP_INFO(this->get_logger(), "Successfully appended TensorRT execution provider options.");
    } catch (const Ort::Exception &e) {
      RCLCPP_ERROR(this->get_logger(), "Error appending TensorRT: %s", e.what());
      execution_provider_ = "cuda";
    }
  }

  if (execution_provider_ == "cuda") {
    OrtCUDAProviderOptions cuda_options{};
    cuda_options.device_id = 0;
    try {
      session_options_.AppendExecutionProvider_CUDA(cuda_options);
      RCLCPP_INFO(this->get_logger(), "Successfully appended CUDA execution provider.");
    } catch (const Ort::Exception &e) {
      RCLCPP_WARN(this->get_logger(), "Falling back to CPU provider.");
      execution_provider_ = "cpu";
    }
  }

  ort_sessions_.clear();

  // LOGIC: If parallel, create 4 sessions. If sequential, create 1 session.
  int num_sessions_to_create = run_parallel_ ? 4 : 1;
  RCLCPP_INFO(this->get_logger(), "Creating %d Inference Session(s)...", num_sessions_to_create);

  for(int i=0; i < num_sessions_to_create; i++) {
      try {
          auto session = std::make_shared<Ort::Session>(
              ort_env_, onnx_model_path_.c_str(), session_options_);
          ort_sessions_.push_back(session);
      } catch (const std::exception &e) {
          RCLCPP_FATAL(this->get_logger(), "Failed to create session %d: %s", i, e.what());
          throw;
      }
  }

  RCLCPP_INFO(this->get_logger(), "Successfully loaded %zu ONNX sessions.", ort_sessions_.size());
}

void DepthEstimation::InitializeCalibrationData() {
  RCLCPP_INFO(this->get_logger(), "Loading calibration data...");
  const std::string calibration_dir = calibration_base_path_ + "/final_maps_" + calibration_resolution_;

  for (const auto &pair_str : stereo_pairs_) {
    std::string map_path = calibration_dir + "/final_rectified_to_fisheye_map_" + pair_str + ".yml";
    std::string config_path = calibration_dir + "/final_map_config_" + pair_str + ".yaml";

    StereoPairData data;
    YAML::Node config = YAML::LoadFile(config_path);
    data.baseline_meters = config["baseline_meters"].as<double>();
    auto resolution = config["final_resolution"].as<std::vector<int>>();
    data.resolution = cv::Size(resolution[0], resolution[1]);
    auto k_values = config["K_rect_left"].as<std::vector<std::vector<double>>>();
    data.K_rect_left = (cv::Mat_<double>(3, 3) << k_values[0][0], k_values[0][1], k_values[0][2],
                        k_values[1][0], k_values[1][1], k_values[1][2], k_values[2][0], k_values[2][1], k_values[2][2]);

    auto T_values = config["T_rect_left_cam0"].as<std::vector<std::vector<double>>>();
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        data.transform_rect_left_to_cam0(i, j) = static_cast<float>(T_values[i][j]);
      }
    }

    cv::FileStorage fs(map_path, cv::FileStorage::READ);
    if (!fs.isOpened()) throw std::runtime_error("Failed to open map file: " + map_path);
    fs["map_left_x"] >> data.map_lx;
    fs["map_left_y"] >> data.map_ly;
    fs["map_right_x"] >> data.map_rx;
    fs["map_right_y"] >> data.map_ry;
    fs.release();

    calibration_data_[pair_str] = data;
  }
}

void DepthEstimation::InitializePublishers() {
  std::string topic_prefix;
  this->get_parameter("output_pointcloud_topic_prefix", topic_prefix);
  for (const auto &pair_str : stereo_pairs_) {
    std::string topic_name = topic_prefix + "/" + "pair_" + pair_str;
    pointcloud_pubs_[pair_str] = this->create_publisher<sensor_msgs::msg::PointCloud2>(topic_name, 10);
  }

  // Initialize Combined Grid Publisher if Verbose is ON
  if (verbose_) {
      debug_grid_pub_ = this->create_publisher<sensor_msgs::msg::CompressedImage>("/depth/debug_grid/compressed", 10);
  }

  if (publish_combined_pointcloud_) {
    combined_pointcloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            combined_pointcloud_topic_, 10);
  }
}

void DepthEstimation::InitializeSubscribers() {
  auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable();

  if (use_compressed_image_) {
      RCLCPP_INFO(this->get_logger(), "Subscribing to COMPRESSED image topic: %s", input_image_topic_.c_str());
      compressed_image_sub_ =
          this->create_subscription<sensor_msgs::msg::CompressedImage>(
              input_image_topic_, qos,
              std::bind(&DepthEstimation::CompressedImageCallback, this,
                        std::placeholders::_1));
  } else {
      RCLCPP_INFO(this->get_logger(), "Subscribing to RAW image topic: %s", input_image_topic_.c_str());
      raw_image_sub_ =
          this->create_subscription<sensor_msgs::msg::Image>(
              input_image_topic_, qos,
              std::bind(&DepthEstimation::RawImageCallback, this,
                        std::placeholders::_1));
  }
}

// --------------------------------------------------------------------------
// IMAGE CALLBACKS & SHARED PROCESSING
// --------------------------------------------------------------------------

void DepthEstimation::CompressedImageCallback(
    const sensor_msgs::msg::CompressedImage::SharedPtr msg) {
  if (!is_initialized_) return;

  auto t_start_total = std::chrono::high_resolution_clock::now();

  cv::Mat decoded_image;
  try {
    decoded_image = cv::imdecode(cv::Mat(msg->data), cv::IMREAD_COLOR);
  } catch (const cv::Exception &e) {
    RCLCPP_ERROR(this->get_logger(), "cv::imdecode error: %s", e.what());
    return;
  }
  if (decoded_image.empty()) return;

  auto t_end_decode = std::chrono::high_resolution_clock::now();
  double ms_decode = std::chrono::duration<double, std::milli>(t_end_decode - t_start_total).count();

  ProcessImage(decoded_image, msg->header, t_start_total, ms_decode);
}

void DepthEstimation::RawImageCallback(
    const sensor_msgs::msg::Image::SharedPtr msg) {
  if (!is_initialized_) return;

  auto t_start_total = std::chrono::high_resolution_clock::now();

  cv_bridge::CvImagePtr cv_ptr;
  try {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  } catch (cv_bridge::Exception& e) {
      RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
      return;
  }

  auto t_end_convert = std::chrono::high_resolution_clock::now();
  double ms_convert = std::chrono::duration<double, std::milli>(t_end_convert - t_start_total).count();

  ProcessImage(cv_ptr->image, msg->header, t_start_total, ms_convert);
}

// --- CORE FUNCTION ---
void DepthEstimation::ProcessImage(const cv::Mat &concatenated_image,
                                   const std_msgs::msg::Header &header,
                                   const std::chrono::high_resolution_clock::time_point &t_start_total,
                                   double preprocessing_ms) {

  if (fisheye_image_width_ == 0) {
    fisheye_image_width_ = concatenated_image.cols / 4;
  }

  // 1. SPLIT
  std::vector<cv::Mat> fisheye_images;
  for (int i = 0; i < 4; ++i) {
    fisheye_images.push_back(concatenated_image(
        cv::Rect(i * fisheye_image_width_, 0, fisheye_image_width_,
                 concatenated_image.rows)));
  }

  auto combined_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

  // Define the worker function locally to avoid duplication
  auto process_pair_task = [&](std::string pair_str, int s_idx) -> ProcessingResult {
      ProcessingResult res;
      res.pair_name = pair_str;
      res.success = false;

      int left_idx = pair_str[0] - '0';
      int right_idx = pair_str[2] - '0';
      const auto &pair_data = calibration_data_.at(pair_str);

      // A. RECTIFY
      auto t_start_rect = std::chrono::high_resolution_clock::now();
      cv::Mat rectified_left, rectified_right;
      cv::remap(fisheye_images[left_idx], rectified_left, pair_data.map_lx,
                pair_data.map_ly, cv::INTER_LINEAR);
      cv::remap(fisheye_images[right_idx], rectified_right, pair_data.map_rx,
                pair_data.map_ry, cv::INTER_LINEAR);
      res.rect_time_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_start_rect).count();

      // B. INFERENCE
      auto t_start_infer = std::chrono::high_resolution_clock::now();
      cv::Mat disparity_map = RunInference(rectified_left, rectified_right, s_idx);
      res.infer_time_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_start_infer).count();

      if (disparity_map.empty()) return res;

      // C. VIZ PREP (Only generated if verbose is active)
      if (verbose_) {
          cv::Mat valid_mask = (disparity_map > 0.0f) & (disparity_map <= 128.0f);
          cv::Mat disparity_normalized;
          cv::normalize(disparity_map, disparity_normalized, 0, 255, cv::NORM_MINMAX, CV_8U, valid_mask);
          cv::applyColorMap(disparity_normalized, res.display_image, cv::COLORMAP_JET);
          cv::Mat display_mask;
          cv::bitwise_not(valid_mask, display_mask);
          res.display_image.setTo(cv::Scalar(0, 0, 0), display_mask);
          cv::putText(res.display_image, "Pair " + pair_str, cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
      }

      // D. POINT CLOUD
      auto t_start_cloud = std::chrono::high_resolution_clock::now();
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = DisparityToPointCloud(disparity_map, pair_data);
      auto cloud_in_cam0 = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
      pcl::transformPointCloud(*cloud, *cloud_in_cam0, pair_data.transform_rect_left_to_cam0);
      auto transformed_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
      pcl::transformPointCloud(*cloud_in_cam0, *transformed_cloud, transform_cam0_to_centroid_);

      // E. PUBLISH / AGGREGATE
      if (publish_combined_pointcloud_) {
          std::lock_guard<std::mutex> lock(combined_cloud_mutex_);
          *combined_cloud += *transformed_cloud;
      } else {
          auto pc_msg = std::make_unique<sensor_msgs::msg::PointCloud2>();
          pcl::toROSMsg(*transformed_cloud, *pc_msg);
          pc_msg->header.stamp = header.stamp;
          pc_msg->header.frame_id = pointcloud_frame_id_;
          pointcloud_pubs_[pair_str]->publish(std::move(pc_msg));
      }

      res.cloud_time_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_start_cloud).count();
      res.success = true;
      return res;
  };

  // 2. EXECUTE (Parallel or Sequential)
  std::vector<ProcessingResult> results;
  std::vector<std::future<ProcessingResult>> futures;

  int pair_idx = 0;
  for (const auto &pair_str : stereo_pairs_) {
      // If parallel, use distinct sessions (idx = 0,1,2,3).
      // If sequential, reuse session 0 (idx = 0).
      int session_idx = run_parallel_ ? pair_idx : 0;

      if (run_parallel_) {
          futures.push_back(std::async(std::launch::async, process_pair_task, pair_str, session_idx));
      } else {
          // Sequential: Run immediately on this thread
          results.push_back(process_pair_task(pair_str, session_idx));
      }
      pair_idx++;
  }

  // If parallel, wait for threads to finish and collect results
  if (run_parallel_) {
      for (auto &f : futures) {
          results.push_back(f.get());
      }
  }

  // 3. REPORT & DEBUG VISUALIZATION
  std::stringstream report;
  std::vector<cv::Mat> grid_images;

  if (verbose_) {
    report << "\n=== PIPELINE REPORT (" << (run_parallel_ ? "PARALLEL" : "SEQUENTIAL") << ") ===\n";
    report << "Pre-process: " << std::fixed << std::setprecision(2) << preprocessing_ms << " ms\n";
  }

  for (const auto &res : results) {
      if (res.success) {
          if (verbose_) {
              report << "  [" << res.pair_name << "] Rect:" << res.rect_time_ms
                     << " | Infer:" << res.infer_time_ms
                     << " | Cloud:" << res.cloud_time_ms << " ms\n";

              if (!res.display_image.empty()) {
                  grid_images.push_back(res.display_image);
              } else {
                  grid_images.push_back(cv::Mat::zeros(cv::Size(224, 224), CV_8UC3));
              }
          }
      } else {
           if (verbose_) {
               report << "  [" << res.pair_name << "] FAILED\n";
               grid_images.push_back(cv::Mat::zeros(cv::Size(224, 224), CV_8UC3));
           }
      }
  }

  // 4. COMBINED GRID PUBLISH (Only if Verbose)
  if (verbose_ && grid_images.size() == 4) {
      cv::Mat top_row, bottom_row, grid_view;
      cv::hconcat(grid_images[0], grid_images[1], top_row);
      cv::hconcat(grid_images[2], grid_images[3], bottom_row);
      cv::vconcat(top_row, bottom_row, grid_view);

      std::vector<uchar> buffer;
      if (cv::imencode(".jpg", grid_view, buffer)) {
          auto msg = std::make_unique<sensor_msgs::msg::CompressedImage>();
          msg->header = header;
          msg->format = "jpeg";
          msg->data = buffer;
          debug_grid_pub_->publish(std::move(msg));
      }
  }

  // 5. COMBINED POINTCLOUD PUBLISH
  auto t_start_comb = std::chrono::high_resolution_clock::now();
  if (publish_combined_pointcloud_) {
    auto combined_pc_msg = std::make_unique<sensor_msgs::msg::PointCloud2>();
    pcl::toROSMsg(*combined_cloud, *combined_pc_msg);
    combined_pc_msg->header.stamp = header.stamp;
    combined_pc_msg->header.frame_id = pointcloud_frame_id_;
    combined_pointcloud_pub_->publish(std::move(combined_pc_msg));
  }
  double ms_comb = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_start_comb).count();

  if (verbose_ && publish_combined_pointcloud_) {
      report << "Combined Publish: " << ms_comb << " ms\n";
  }

  auto t_end_total = std::chrono::high_resolution_clock::now();
  double ms_total = std::chrono::duration<double, std::milli>(t_end_total - t_start_total).count();

  if (verbose_) {
      report << "--------------------------------\n";
      report << "TOTAL WALL TIME: " << ms_total << " ms (" << (1000.0/ms_total) << " FPS)";
      RCLCPP_INFO(this->get_logger(), "%s", report.str().c_str());
  }
}

cv::Mat DepthEstimation::RunInference(const cv::Mat &rectified_left,
                                      const cv::Mat &rectified_right,
                                      int session_idx) {
  std::vector<const char *> input_names_char;
  input_names_char.reserve(input_node_names_.size());
  for (const auto &name : input_node_names_) input_names_char.push_back(name.c_str());

  std::vector<const char *> output_names_char;
  output_names_char.reserve(output_node_names_.size());
  for (const auto &name : output_node_names_) output_names_char.push_back(name.c_str());

  std::vector<Ort::Value> input_tensors;
  Ort::AllocatorWithDefaultOptions allocator;

  if (model_type_ == "s2m2") {
    cv::Mat blob_left = cv::dnn::blobFromImage(rectified_left, 1.0, cv::Size(model_full_width_, model_full_height_), cv::Scalar(), true, false, CV_8U);
    cv::Mat blob_right = cv::dnn::blobFromImage(rectified_right, 1.0, cv::Size(model_full_width_, model_full_height_), cv::Scalar(), true, false, CV_8U);

    input_tensors.push_back(Ort::Value::CreateTensor<uint8_t>(allocator.GetInfo(), blob_left.ptr<uint8_t>(), blob_left.total(), input_node_dims_[0].data(), input_node_dims_[0].size()));
    input_tensors.push_back(Ort::Value::CreateTensor<uint8_t>(allocator.GetInfo(), blob_right.ptr<uint8_t>(), blob_right.total(), input_node_dims_[1].data(), input_node_dims_[1].size()));

  } else if (model_type_ == "hitnet") {
    cv::Mat left_gray, right_gray;
    if (rectified_left.channels() == 3) {
      cv::cvtColor(rectified_left, left_gray, cv::COLOR_BGR2GRAY);
      cv::cvtColor(rectified_right, right_gray, cv::COLOR_BGR2GRAY);
    } else {
      left_gray = rectified_left;
      right_gray = rectified_right;
    }
    cv::Mat left_resized, right_resized;
    cv::resize(left_gray, left_resized, cv::Size(model_full_width_, model_full_height_));
    cv::resize(right_gray, right_resized, cv::Size(model_full_width_, model_full_height_));
    left_resized.convertTo(left_resized, CV_32F, 1.0 / 255.0);
    right_resized.convertTo(right_resized, CV_32F, 1.0 / 255.0);
    std::vector<cv::Mat> images = {left_resized, right_resized};
    cv::Mat blob = cv::dnn::blobFromImages(images, 1.0, cv::Size(), cv::Scalar(), false);
    std::vector<int64_t> input_shape = {1, 2, model_full_height_, model_full_width_};
    input_tensors.push_back(Ort::Value::CreateTensor<float>(allocator.GetInfo(), blob.ptr<float>(), blob.total(), input_shape.data(), input_shape.size()));

  } else if (model_type_ == "crestereo") {
    cv::Mat left_bgr, right_bgr;
    if (rectified_left.channels() == 1) {
        cv::cvtColor(rectified_left, left_bgr, cv::COLOR_GRAY2BGR);
        cv::cvtColor(rectified_right, right_bgr, cv::COLOR_GRAY2BGR);
    } else {
        left_bgr = rectified_left;
        right_bgr = rectified_right;
    }
    left_bgr.convertTo(left_bgr, CV_32F);
    right_bgr.convertTo(right_bgr, CV_32F);

    if (input_node_names_.size() == 4) { // 4-Input
      cv::Mat left_full_resized, right_full_resized, left_half_resized, right_half_resized;
      cv::resize(left_bgr, left_full_resized, cv::Size(model_full_width_, model_full_height_));
      cv::resize(right_bgr, right_full_resized, cv::Size(model_full_width_, model_full_height_));
      cv::resize(left_bgr, left_half_resized, cv::Size(model_half_width_, model_half_height_));
      cv::resize(right_bgr, right_half_resized, cv::Size(model_half_width_, model_half_height_));

      cv::Mat blob_left = cv::dnn::blobFromImage(left_full_resized);
      cv::Mat blob_right = cv::dnn::blobFromImage(right_full_resized);
      cv::Mat blob_left_half = cv::dnn::blobFromImage(left_half_resized);
      cv::Mat blob_right_half = cv::dnn::blobFromImage(right_half_resized);

      input_tensors.push_back(Ort::Value::CreateTensor<float>(allocator.GetInfo(), blob_left_half.ptr<float>(), blob_left_half.total(), input_node_dims_[0].data(), input_node_dims_[0].size()));
      input_tensors.push_back(Ort::Value::CreateTensor<float>(allocator.GetInfo(), blob_right_half.ptr<float>(), blob_right_half.total(), input_node_dims_[1].data(), input_node_dims_[1].size()));
      input_tensors.push_back(Ort::Value::CreateTensor<float>(allocator.GetInfo(), blob_left.ptr<float>(), blob_left.total(), input_node_dims_[2].data(), input_node_dims_[2].size()));
      input_tensors.push_back(Ort::Value::CreateTensor<float>(allocator.GetInfo(), blob_right.ptr<float>(), blob_right.total(), input_node_dims_[3].data(), input_node_dims_[3].size()));

    } else if (input_node_names_.size() == 2) { // 2-Input
      cv::Mat left_resized, right_resized;
      cv::resize(left_bgr, left_resized, cv::Size(model_full_width_, model_full_height_));
      cv::resize(right_bgr, right_resized, cv::Size(model_full_width_, model_full_height_));
      cv::Mat blob_left = cv::dnn::blobFromImage(left_resized);
      cv::Mat blob_right = cv::dnn::blobFromImage(right_resized);
      input_tensors.push_back(Ort::Value::CreateTensor<float>(allocator.GetInfo(), blob_left.ptr<float>(), blob_left.total(), input_node_dims_[0].data(), input_node_dims_[0].size()));
      input_tensors.push_back(Ort::Value::CreateTensor<float>(allocator.GetInfo(), blob_right.ptr<float>(), blob_right.total(), input_node_dims_[1].data(), input_node_dims_[1].size()));
    }
  }

  auto output_tensors = ort_sessions_[session_idx]->Run(
      Ort::RunOptions{nullptr},
      input_names_char.data(), input_tensors.data(), input_tensors.size(),
      output_names_char.data(), output_names_char.size());

  cv::Mat disparity_map;
  if (model_type_ == "hitnet") {
    int original_height = rectified_left.rows;
    int original_width = rectified_left.cols;
    float *float_buffer = output_tensors[0].GetTensorMutableData<float>();
    cv::Mat disparity_small(model_full_height_, model_full_width_, CV_32F, float_buffer);
    cv::resize(disparity_small, disparity_map, cv::Size(original_width, original_height), 0, 0, cv::INTER_LINEAR);
    disparity_map *= (static_cast<float>(original_width) / static_cast<float>(model_full_width_));

  } else if (model_type_ == "s2m2") {
    auto type_info = output_tensors[0].GetTensorTypeAndShapeInfo();
    int out_h = type_info.GetShape()[2];
    int out_w = type_info.GetShape()[3];
    float *disp_ptr = output_tensors[0].GetTensorMutableData<float>();
    cv::Mat disp_small(out_h, out_w, CV_32F, disp_ptr);
    cv::resize(disp_small, disparity_map, cv::Size(rectified_left.cols, rectified_left.rows), 0, 0, cv::INTER_LINEAR);
    disparity_map *= (static_cast<float>(rectified_left.cols) / static_cast<float>(out_w));

    if (filter_disparity_ && output_tensors.size() >= 3) {
        float *occ_ptr = output_tensors[1].GetTensorMutableData<float>();
        float *conf_ptr = output_tensors[2].GetTensorMutableData<float>();
        cv::Mat occ_small(out_h, out_w, CV_32F, occ_ptr);
        cv::Mat conf_small(out_h, out_w, CV_32F, conf_ptr);
        cv::Mat occ_map, conf_map;
        cv::resize(occ_small, occ_map, cv::Size(rectified_left.cols, rectified_left.rows), 0, 0, cv::INTER_LINEAR);
        cv::resize(conf_small, conf_map, cv::Size(rectified_left.cols, rectified_left.rows), 0, 0, cv::INTER_LINEAR);

        for (int i = 0; i < disparity_map.rows; ++i) {
            float* d_row = disparity_map.ptr<float>(i);
            const float* o_row = occ_map.ptr<float>(i);
            const float* c_row = conf_map.ptr<float>(i);
            for (int j = 0; j < disparity_map.cols; ++j) {
                if (c_row[j] <= 0.1f || o_row[j] <= 0.01f) d_row[j] = 0.0f;
            }
        }
    }
  } else if (model_type_ == "crestereo") {
    auto shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    float *float_buffer = output_tensors[0].GetTensorMutableData<float>();
    cv::Mat disp(shape[2], shape[3], CV_32F);
    memcpy(disp.data, float_buffer, shape[2] * shape[3] * sizeof(float));
    disparity_map = disp;
  }
  return disparity_map;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr
DepthEstimation::DisparityToPointCloud(const cv::Mat &disparity_map, const StereoPairData &pair_data) {
  auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  cloud->width = disparity_map.cols;
  cloud->height = disparity_map.rows;
  cloud->is_dense = false;
  cloud->points.resize(cloud->width * cloud->height);

  const double fx = pair_data.K_rect_left.at<double>(0, 0);
  const double fy = pair_data.K_rect_left.at<double>(1, 1);
  const double cx = pair_data.K_rect_left.at<double>(0, 2);
  const double cy = pair_data.K_rect_left.at<double>(1, 2);
  const double baseline = pair_data.baseline_meters;

  for (int v = 0; v < disparity_map.rows; ++v) {
    for (int u = 0; u < disparity_map.cols; ++u) {
      float disparity = disparity_map.at<float>(v, u);
      pcl::PointXYZ &point = cloud->at(u, v);
      if (disparity > 0.0f) {
        double Z = (fx * baseline) / disparity;
        point.x = (u - cx) * Z / fx;
        point.y = (v - cy) * Z / fy;
        point.z = Z;
      } else {
        point.x = point.y = point.z = std::numeric_limits<float>::quiet_NaN();
      }
    }
  }
  return cloud;
}

void DepthEstimation::InitializeServices() {
  camera_info_service_ = this->create_service<depth_estimation_ros2::srv::GetCameraInfo>(
          "~/get_camera_info",
          std::bind(&DepthEstimation::GetCameraInfoCallback, this, std::placeholders::_1, std::placeholders::_2));
  RCLCPP_INFO(this->get_logger(), "Service 'get_camera_info' is ready.");
}

void DepthEstimation::GetCameraInfoCallback(
    const std::shared_ptr<depth_estimation_ros2::srv::GetCameraInfo::Request> request,
    std::shared_ptr<depth_estimation_ros2::srv::GetCameraInfo::Response> response) {
  (void)request;
  if (calibration_data_.empty()) return;

  std::map<int, bool> processed_cameras;
  for (const auto &pair_str : stereo_pairs_) {
    int cam_idx = pair_str[0] - '0';
    if (processed_cameras.find(cam_idx) == processed_cameras.end()) {
      const auto &pair_data = calibration_data_.at(pair_str);
      depth_estimation_ros2::msg::CameraInfo cam_info;
      cam_info.camera_id = "rectified_cam_" + std::to_string(cam_idx);
      Eigen::Matrix4f transform_rect_cam_to_centroid = transform_cam0_to_centroid_ * pair_data.transform_rect_left_to_cam0;
      Eigen::Isometry3f isometry(transform_rect_cam_to_centroid);
      cam_info.pose = tf2::toMsg(isometry.cast<double>());
      double fx = pair_data.K_rect_left.at<double>(0, 0);
      double width = static_cast<double>(pair_data.resolution.width);
      double height = static_cast<double>(pair_data.resolution.height);
      cam_info.hfov_radians = 2.0 * std::atan(width / (2.0 * fx));
      double fy = pair_data.K_rect_left.at<double>(1, 1);
      cam_info.vfov_radians = 2.0 * std::atan(height / (2.0 * fy));
      response->cameras.push_back(cam_info);
      processed_cameras[cam_idx] = true;
    }
  }
}

} // namespace depth_estimation
