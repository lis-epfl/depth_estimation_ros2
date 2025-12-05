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
#include <thread>

namespace depth_estimation {

DepthEstimation::DepthEstimation(const rclcpp::NodeOptions &options)
    : Node("depth_estimation_node", options),
      ort_env_(ORT_LOGGING_LEVEL_WARNING, "depth_estimation") {

  RCLCPP_INFO(this->get_logger(), "Initializing Depth Estimation Node (Standard Mode)...");

  DeclareRosParameters();
  InitializeRosParameters();

  try {
    InitializeModel();
    InitializeCalibrationData();
    InitializeTransform();
    InitializeLogging();

    // Allocate memory ONCE at startup to prevent runtime alloc spikes
    AllocateWorkspaces();

  } catch (const std::exception &e) {
    RCLCPP_FATAL(this->get_logger(), "Initialization failed: %s", e.what());
    rclcpp::shutdown();
    return;
  }

  InitializePublishers();
  InitializeSubscribers();
  InitializeServices();

  is_initialized_ = true;
  RCLCPP_INFO(this->get_logger(), "Node ready.");
}

DepthEstimation::~DepthEstimation() {
  if (timing_file_.is_open()) {
    timing_file_.close();
  }
  RCLCPP_INFO(this->get_logger(), "Shutting down Depth Estimation Node.");
}

// ---------------------------------------------------------
// ALLOCATE WORKSPACES
// ---------------------------------------------------------
void DepthEstimation::AllocateWorkspaces() {
    workspaces_.resize(stereo_pairs_.size());
    RCLCPP_INFO(this->get_logger(), "Pre-allocating memory for %zu stereo pairs...", stereo_pairs_.size());

    int blob_w = model_full_width_;
    int blob_h = model_full_height_;

    for (auto &ws : workspaces_) {
        // Init Rect buffers (used as input to inference)
        ws.rect_left = cv::Mat::zeros(blob_h, blob_w, CV_8UC3);
        ws.rect_right = cv::Mat::zeros(blob_h, blob_w, CV_8UC3);

        if (model_type_ == "hitnet" || model_type_ == "crestereo") {
            ws.blob_left = cv::Mat(blob_h, blob_w, CV_32FC1);
            ws.blob_right = cv::Mat(blob_h, blob_w, CV_32FC1);
        } else {
             // S2M2 uses [1, 3, H, W] CV_8U
             int sz[] = {1, 3, blob_h, blob_w};
             ws.blob_left = cv::Mat(4, sz, CV_8U);
             ws.blob_right = cv::Mat(4, sz, CV_8U);
        }

        ws.disparity = cv::Mat(blob_h, blob_w, CV_32FC1);
        ws.cloud_msg->points.reserve(blob_w * blob_h);
    }
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
  this->declare_parameter<bool>("run_parallel", true);
  this->declare_parameter<bool>("use_compressed_image", true);
  this->declare_parameter<std::string>("input_image_topic", "/oak_ffc_4p_driver_node/compressed");
  this->declare_parameter<std::string>("transform_config_path", "");
  this->declare_parameter<std::string>("onnx_model_path", "model.onnx");
  this->declare_parameter<std::string>("model_type", "hitnet");
  this->declare_parameter<std::string>("execution_provider", "tensorrt");
  this->declare_parameter<std::string>("output_pointcloud_topic_prefix", "/depth/pointcloud");
  this->declare_parameter<std::vector<std::string>>("stereo_pairs", {"0_1", "1_2", "2_3", "3_0"});
  this->declare_parameter<std::string>("calibration_base_path", "");
  this->declare_parameter<std::string>("calibration_resolution", "120_160");
  this->declare_parameter<bool>("publish_combined_pointcloud", false);
  this->declare_parameter<std::string>("combined_pointcloud_topic", "/depth/pointcloud/combined");
  this->declare_parameter<std::string>("pointcloud_frame_id", "drone_centroid");
  this->declare_parameter<bool>("filter_disparity", false);
  this->declare_parameter<bool>("logging.enabled", true);
  this->declare_parameter<std::string>("logging.directory", "/tmp/depth_logs");
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
  this->get_parameter("publish_combined_pointcloud", publish_combined_pointcloud_);
  this->get_parameter("combined_pointcloud_topic", combined_pointcloud_topic_);
  this->get_parameter("pointcloud_frame_id", pointcloud_frame_id_);
  this->get_parameter("filter_disparity", filter_disparity_);
  this->get_parameter("logging.enabled", enable_logging_);
  this->get_parameter("logging.directory", logging_directory_);

  if (calibration_base_path_.empty()) throw std::runtime_error("Calibration path empty");
}

void DepthEstimation::InitializeLogging() {
  if (!enable_logging_) return;
  if (!std::filesystem::exists(logging_directory_)) {
      std::filesystem::create_directories(logging_directory_);
  }
  std::string filename = logging_directory_ + "/depth_computation_times.csv";
  timing_file_.open(filename);
  if (timing_file_.is_open()) {
     timing_file_ << "timestamp,preprocess_ms";
     for (const auto& pair : stereo_pairs_) {
       timing_file_ << "," << pair << "_rect_ms," << pair << "_infer_ms," << pair << "_cloud_ms";
     }
     timing_file_ << ",combined_publish_ms,total_wall_ms\n";
  }
}

void DepthEstimation::LogTiming(double timestamp, double preprocess_ms,
                                const std::vector<ProcessingResult>& results,
                                double combined_publish_ms, double total_ms) {
  if (!timing_file_.is_open()) return;
  timing_file_ << std::fixed << std::setprecision(6) << timestamp << "," << std::setprecision(3) << preprocess_ms;
  for (const auto& res : results) {
    timing_file_ << "," << res.rect_time_ms << "," << res.infer_time_ms << "," << res.cloud_time_ms;
  }
  timing_file_ << "," << combined_publish_ms << "," << total_ms << "\n";
  timing_file_.flush();
}

void DepthEstimation::InitializeTransform() {
  YAML::Node config = YAML::LoadFile(transform_config_path_);
  const auto &data = config["transform"]["data"].as<std::vector<float>>();
  transform_cam0_to_centroid_ = Eigen::Map<const Eigen::Matrix<float, 4, 4, Eigen::RowMajor>>(data.data());
}

void DepthEstimation::InitializeModel() {
  Ort::AllocatorWithDefaultOptions allocator;
  Ort::SessionOptions inspect_options;
  Ort::Session inspect_session(ort_env_, onnx_model_path_.c_str(), inspect_options);

  size_t num_input_nodes = inspect_session.GetInputCount();
  input_node_names_.clear();
  input_node_dims_.clear();
  for (size_t i = 0; i < num_input_nodes; i++) {
    auto input_name_ptr = inspect_session.GetInputNameAllocated(i, allocator);
    input_node_names_.emplace_back(input_name_ptr.get());
    input_node_dims_.push_back(inspect_session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }

  size_t num_output_nodes = inspect_session.GetOutputCount();
  output_node_names_.clear();
  for (size_t i = 0; i < num_output_nodes; i++) {
    auto output_name_ptr = inspect_session.GetOutputNameAllocated(i, allocator);
    output_node_names_.emplace_back(output_name_ptr.get());
  }

  if (model_type_ == "hitnet") {
    model_full_height_ = input_node_dims_[0][2];
    model_full_width_ = input_node_dims_[0][3];
  } else if (model_type_ == "s2m2") {
    model_full_height_ = input_node_dims_[0][2];
    model_full_width_ = input_node_dims_[0][3];
  } else if (model_type_ == "crestereo") {
     if (input_node_dims_.size() > 0) {
        model_full_height_ = input_node_dims_[0][2];
        model_full_width_ = input_node_dims_[0][3];
     }
  }

  session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

  if (execution_provider_ == "tensorrt") {
    OrtTensorRTProviderOptions trt_options{};
    trt_options.device_id = 0;
    trt_options.trt_fp16_enable = 1;
    trt_options.trt_engine_cache_enable = 1;

    std::string arch = GetSystemArchitecture();
    std::string gpu_name = GetGpuName();
    std::filesystem::path onnx_path(onnx_model_path_);
    std::filesystem::path model_dir = onnx_path.parent_path();
    std::string stem = onnx_path.stem().string();

    std::string cache_dir_name = stem + "_" + gpu_name + "_" + arch + "_trt_engine";
    std::filesystem::path cache_path = model_dir / cache_dir_name;

    RCLCPP_INFO(this->get_logger(), "TRT Cache Target: %s", cache_path.string().c_str());

    if (!std::filesystem::exists(cache_path)) {
        std::filesystem::create_directories(cache_path);
    }

    static std::string cache_path_str;
    cache_path_str = cache_path.string();
    trt_options.trt_engine_cache_path = cache_path_str.c_str();

    session_options_.AppendExecutionProvider_TensorRT(trt_options);
  } else if (execution_provider_ == "cuda") {
    OrtCUDAProviderOptions cuda_options{};
    session_options_.AppendExecutionProvider_CUDA(cuda_options);
  }

  ort_sessions_.clear();
  int num_sessions = run_parallel_ ? 4 : 1;
  RCLCPP_INFO(this->get_logger(), "Creating %d Inference Session(s)...", num_sessions);
  for(int i=0; i < num_sessions; i++) {
      ort_sessions_.push_back(std::make_shared<Ort::Session>(ort_env_, onnx_model_path_.c_str(), session_options_));
  }
}

void DepthEstimation::InitializeCalibrationData() {
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
        for (int j = 0; j < 4; ++j) data.transform_rect_left_to_cam0(i, j) = static_cast<float>(T_values[i][j]);
    }

    cv::FileStorage fs(map_path, cv::FileStorage::READ);
    fs["map_left_x"] >> data.map_lx;
    fs["map_left_y"] >> data.map_ly;
    fs["map_right_x"] >> data.map_rx;
    fs["map_right_y"] >> data.map_ry;
    calibration_data_[pair_str] = data;
  }
}

void DepthEstimation::InitializePublishers() {
  std::string topic_prefix;
  this->get_parameter("output_pointcloud_topic_prefix", topic_prefix);
  for (const auto &pair_str : stereo_pairs_) {
    pointcloud_pubs_[pair_str] = this->create_publisher<sensor_msgs::msg::PointCloud2>(topic_prefix + "/pair_" + pair_str, 10);
  }
  if (verbose_) debug_grid_pub_ = this->create_publisher<sensor_msgs::msg::CompressedImage>("/depth/debug_grid/compressed", 10);
  if (publish_combined_pointcloud_) combined_pointcloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(combined_pointcloud_topic_, 10);
}

void DepthEstimation::InitializeSubscribers() {
  auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable();
  if (use_compressed_image_) {
      compressed_image_sub_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
          input_image_topic_, qos, std::bind(&DepthEstimation::CompressedImageCallback, this, std::placeholders::_1));
  } else {
      raw_image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
          input_image_topic_, qos, std::bind(&DepthEstimation::RawImageCallback, this, std::placeholders::_1));
  }
}

void DepthEstimation::CompressedImageCallback(const sensor_msgs::msg::CompressedImage::SharedPtr msg) {
  if (!is_initialized_) return;
  auto t_start = std::chrono::high_resolution_clock::now();
  cv::Mat decoded = cv::imdecode(cv::Mat(msg->data), cv::IMREAD_COLOR);
  if (decoded.empty()) return;
  ProcessImage(decoded, msg->header, t_start, 0.0);
}

void DepthEstimation::RawImageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
  if (!is_initialized_) return;
  auto t_start = std::chrono::high_resolution_clock::now();
  cv_bridge::CvImagePtr cv_ptr;
  try { cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8); } catch (...) { return; }
  ProcessImage(cv_ptr->image, msg->header, t_start, 0.0);
}

void DepthEstimation::ProcessImage(const cv::Mat &concatenated_image,
                                   const std_msgs::msg::Header &header,
                                   const std::chrono::high_resolution_clock::time_point &t_start_total,
                                   double preprocessing_ms) {

  if (fisheye_image_width_ == 0) fisheye_image_width_ = concatenated_image.cols / 4;

  auto process_pair_task = [&](int idx) -> ProcessingResult {
      std::string pair_str = stereo_pairs_[idx];
      StereoWorkspace& ws = workspaces_[idx];
      const auto &pair_data = calibration_data_.at(pair_str);
      ProcessingResult res;
      res.pair_name = pair_str;
      res.success = false;

      int left_idx = pair_str[0] - '0';
      int right_idx = pair_str[2] - '0';

      cv::Mat src_left = concatenated_image(cv::Rect(left_idx * fisheye_image_width_, 0, fisheye_image_width_, concatenated_image.rows));
      cv::Mat src_right = concatenated_image(cv::Rect(right_idx * fisheye_image_width_, 0, fisheye_image_width_, concatenated_image.rows));

      auto t_rect = std::chrono::high_resolution_clock::now();
      cv::remap(src_left, ws.rect_left, pair_data.map_lx, pair_data.map_ly, cv::INTER_LINEAR);
      cv::remap(src_right, ws.rect_right, pair_data.map_lx, pair_data.map_ly, cv::INTER_LINEAR);
      res.rect_time_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_rect).count();

      auto t_infer = std::chrono::high_resolution_clock::now();
      RunInference(ws, run_parallel_ ? idx : 0);
      res.infer_time_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_infer).count();

      auto t_cloud = std::chrono::high_resolution_clock::now();
      DisparityToPointCloud(ws.disparity, pair_data, ws.cloud_msg);

      if (!publish_combined_pointcloud_) {
           auto pc_msg = std::make_unique<sensor_msgs::msg::PointCloud2>();
           pcl::toROSMsg(*ws.cloud_msg, *pc_msg);
           pc_msg->header.stamp = header.stamp;
           pc_msg->header.frame_id = pointcloud_frame_id_;
           pointcloud_pubs_[pair_str]->publish(std::move(pc_msg));
      }
      res.cloud_time_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_cloud).count();
      res.success = true;
      return res;
  };

  std::vector<ProcessingResult> results;
  std::vector<std::future<ProcessingResult>> futures;

  for (size_t i = 0; i < stereo_pairs_.size(); ++i) {
      if (run_parallel_) {
          futures.push_back(std::async(std::launch::async, process_pair_task, i));
      } else {
          results.push_back(process_pair_task(i));
      }
  }

  if (run_parallel_) {
      for (auto &f : futures) results.push_back(f.get());
  }

  if (verbose_) {
      auto t_end_total = std::chrono::high_resolution_clock::now();
      double ms_total = std::chrono::duration<double, std::milli>(t_end_total - t_start_total).count();

      std::stringstream report;
      report << "\n--- FRAME STATS ---\n";
      report << "Pre-process: " << std::fixed << std::setprecision(2) << preprocessing_ms << " ms\n";

      for (const auto &res : results) {
          if (res.success) {
              report << " [" << res.pair_name << "] "
                     << "Rect:" << std::setw(4) << res.rect_time_ms << " | "
                     << "Infer:" << std::setw(4) << res.infer_time_ms << " | "
                     << "Cloud:" << std::setw(4) << res.cloud_time_ms << " ms\n";
          }
      }
      report << "TOTAL WALL TIME: " << ms_total << " ms (" << (1000.0/ms_total) << " FPS)";
      RCLCPP_INFO(this->get_logger(), "%s", report.str().c_str());
  }
}

void DepthEstimation::RunInference(StereoWorkspace& ws, int session_idx) {
  Ort::AllocatorWithDefaultOptions allocator;
  std::vector<Ort::Value> input_tensors;

  std::vector<const char *> input_names;
  for(const auto& s : input_node_names_) input_names.push_back(s.c_str());
  std::vector<const char *> output_names;
  for(const auto& s : output_node_names_) output_names.push_back(s.c_str());

  if (model_type_ == "s2m2") {
      cv::dnn::blobFromImage(ws.rect_left, ws.blob_left, 1.0, cv::Size(model_full_width_, model_full_height_), cv::Scalar(), true, false, CV_8U);
      cv::dnn::blobFromImage(ws.rect_right, ws.blob_right, 1.0, cv::Size(model_full_width_, model_full_height_), cv::Scalar(), true, false, CV_8U);

      input_tensors.push_back(Ort::Value::CreateTensor<uint8_t>(allocator.GetInfo(),
          ws.blob_left.ptr<uint8_t>(), ws.blob_left.total(), input_node_dims_[0].data(), input_node_dims_[0].size()));
      input_tensors.push_back(Ort::Value::CreateTensor<uint8_t>(allocator.GetInfo(),
          ws.blob_right.ptr<uint8_t>(), ws.blob_right.total(), input_node_dims_[1].data(), input_node_dims_[1].size()));

  } else {
      if (ws.rect_left.channels() == 3) {
          cv::cvtColor(ws.rect_left, ws.gray_left, cv::COLOR_BGR2GRAY);
          cv::cvtColor(ws.rect_right, ws.gray_right, cv::COLOR_BGR2GRAY);
      } else {
          ws.rect_left.copyTo(ws.gray_left);
          ws.rect_right.copyTo(ws.gray_right);
      }

      cv::dnn::blobFromImage(ws.gray_left, ws.blob_left, 1.0/255.0, cv::Size(model_full_width_, model_full_height_), cv::Scalar(), false, false, CV_32F);
      cv::dnn::blobFromImage(ws.gray_right, ws.blob_right, 1.0/255.0, cv::Size(model_full_width_, model_full_height_), cv::Scalar(), false, false, CV_32F);

      if (model_type_ == "hitnet") {
           if (input_node_names_.size() == 2) {
               input_tensors.push_back(Ort::Value::CreateTensor<float>(allocator.GetInfo(), ws.blob_left.ptr<float>(), ws.blob_left.total(), input_node_dims_[0].data(), input_node_dims_[0].size()));
               input_tensors.push_back(Ort::Value::CreateTensor<float>(allocator.GetInfo(), ws.blob_right.ptr<float>(), ws.blob_right.total(), input_node_dims_[1].data(), input_node_dims_[1].size()));
           } else {
               std::vector<cv::Mat> gray_srcs = {ws.gray_left, ws.gray_right};
               cv::Mat combined = cv::dnn::blobFromImages(gray_srcs, 1.0/255.0, cv::Size(model_full_width_, model_full_height_), cv::Scalar(), false, false, CV_32F);
               input_tensors.push_back(Ort::Value::CreateTensor<float>(allocator.GetInfo(), combined.ptr<float>(), combined.total(), input_node_dims_[0].data(), input_node_dims_[0].size()));
           }
       } else {
           input_tensors.push_back(Ort::Value::CreateTensor<float>(allocator.GetInfo(), ws.blob_left.ptr<float>(), ws.blob_left.total(), input_node_dims_[0].data(), input_node_dims_[0].size()));
           input_tensors.push_back(Ort::Value::CreateTensor<float>(allocator.GetInfo(), ws.blob_right.ptr<float>(), ws.blob_right.total(), input_node_dims_[1].data(), input_node_dims_[1].size()));
       }
  }

  auto output_tensors = ort_sessions_[session_idx]->Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), input_tensors.size(), output_names.data(), output_names.size());

  float *raw_ptr = output_tensors[0].GetTensorMutableData<float>();
  auto type_info = output_tensors[0].GetTensorTypeAndShapeInfo();
  int out_h = type_info.GetShape()[2];
  int out_w = type_info.GetShape()[3];

  cv::Mat output_wrapper(out_h, out_w, CV_32F, raw_ptr);
  cv::resize(output_wrapper, ws.disparity, cv::Size(ws.rect_left.cols, ws.rect_left.rows));
}

void DepthEstimation::DisparityToPointCloud(const cv::Mat &disparity_map, const StereoPairData &pair_data, pcl::PointCloud<pcl::PointXYZ>::Ptr &out_cloud) {
  out_cloud->clear();
  out_cloud->width = disparity_map.cols;
  out_cloud->height = disparity_map.rows;
  out_cloud->is_dense = false;

  if (out_cloud->points.capacity() < out_cloud->width * out_cloud->height) {
      out_cloud->points.reserve(out_cloud->width * out_cloud->height);
  }
  out_cloud->points.resize(out_cloud->width * out_cloud->height);

  const double fx = pair_data.K_rect_left.at<double>(0, 0);
  const double fy = pair_data.K_rect_left.at<double>(1, 1);
  const double cx = pair_data.K_rect_left.at<double>(0, 2);
  const double cy = pair_data.K_rect_left.at<double>(1, 2);
  const double baseline = pair_data.baseline_meters;

  const float* disp_ptr = (const float*)disparity_map.data;
  pcl::PointXYZ* cloud_ptr = &out_cloud->points[0];
  int total = disparity_map.rows * disparity_map.cols;

  for(int i=0; i<total; ++i) {
      float d = disp_ptr[i];
      if (d > 0.0f) {
          int u = i % disparity_map.cols;
          int v = i / disparity_map.cols;
          double Z = (fx * baseline) / d;
          cloud_ptr[i].x = (u - cx) * Z / fx;
          cloud_ptr[i].y = (v - cy) * Z / fy;
          cloud_ptr[i].z = Z;
      } else {
          cloud_ptr[i].x = cloud_ptr[i].y = cloud_ptr[i].z = std::numeric_limits<float>::quiet_NaN();
      }
  }
}

void DepthEstimation::InitializeServices() {
  camera_info_service_ = this->create_service<depth_estimation_ros2::srv::GetCameraInfo>(
          "~/get_camera_info",
          std::bind(&DepthEstimation::GetCameraInfoCallback, this, std::placeholders::_1, std::placeholders::_2));
  RCLCPP_INFO(this->get_logger(), "Service 'get_camera_info' is ready.");
}

void DepthEstimation::GetCameraInfoCallback(const std::shared_ptr<depth_estimation_ros2::srv::GetCameraInfo::Request> req, std::shared_ptr<depth_estimation_ros2::srv::GetCameraInfo::Response> res) {
   (void)req;
   if (calibration_data_.empty()) return;
   std::map<int, bool> processed_cameras;
   for (const auto &pair_str : stereo_pairs_) {
     int cam_idx = pair_str[0] - '0';
     if (processed_cameras.find(cam_idx) == processed_cameras.end()) {
        const auto &pair_data = calibration_data_.at(pair_str);
        depth_estimation_ros2::msg::CameraInfo cam_info;
        cam_info.camera_id = "rectified_cam_" + std::to_string(cam_idx);
        Eigen::Matrix4f tr = transform_cam0_to_centroid_ * pair_data.transform_rect_left_to_cam0;
        Eigen::Isometry3f iso(tr);
        cam_info.pose = tf2::toMsg(iso.cast<double>());
        double fx = pair_data.K_rect_left.at<double>(0, 0);
        double w = static_cast<double>(pair_data.resolution.width);
        double h = static_cast<double>(pair_data.resolution.height);
        cam_info.hfov_radians = 2.0 * std::atan(w / (2.0 * fx));
        double fy = pair_data.K_rect_left.at<double>(1, 1);
        cam_info.vfov_radians = 2.0 * std::atan(h / (2.0 * fy));
        res->cameras.push_back(cam_info);
        processed_cameras[cam_idx] = true;
     }
   }
}

} // namespace depth_estimation
