#ifndef DEPTH_ESTIMATION_HPP_
#define DEPTH_ESTIMATION_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_eigen/tf2_eigen.hpp>

#include <onnxruntime_cxx_api.h>

#include "depth_estimation_ros2/srv/get_camera_info.hpp"

// --- SYSTEM INCLUDES ---
#include <filesystem>
#include <sys/utsname.h>
#include <cuda_runtime.h>
#include <future>
#include <mutex>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <queue>
#include <condition_variable>
#include <atomic>
#include <thread>

namespace depth_estimation {

struct StereoPairData {
  cv::Size resolution;
  double baseline_meters;
  cv::Mat K_rect_left;
  cv::Mat map_lx, map_ly, map_rx, map_ry;
  Eigen::Matrix4f transform_rect_left_to_cam0;

  // --- DEPTH CORRECTION PARAMETERS ---
  double baseline_scale;      // Multiplier for baseline (default: 1.0)
  double disparity_offset;    // Pixels to subtract from disparity (default: 0.0)
};

// Struct to hold results from inference (main thread)
struct ProcessingResult {
    std::string pair_name;
    double rect_time_ms;
    double infer_time_ms;
    bool success;
};

// Struct to hold data for async pointcloud processing
struct DisparityPayload {
    cv::Mat disparity_map;
    cv::Mat rectified_left;  // For visualization
    std::string pair_name;
    std_msgs::msg::Header header;
    uint64_t frame_id;
    bool verbose;  // Whether to generate debug visualization
    size_t pair_index;  // Index of this pair (0-3) for combined cloud offset

    // Calibration data (copied for thread safety)
    cv::Size resolution;
    double baseline_meters;
    cv::Mat K_rect_left;
    Eigen::Matrix4f transform_rect_left_to_cam0;

    // --- DEPTH CORRECTION PARAMETERS ---
    double baseline_scale;      // Multiplier for baseline
    double disparity_offset;    // Pixels to subtract from disparity
};

// Struct for aggregating combined pointcloud per frame
struct FrameCloudAggregator {
    pcl::PointCloud<pcl::PointXYZ>::Ptr combined_cloud;
    std_msgs::msg::Header header;
    std::atomic<int> pairs_received{0};
    std::mutex cloud_mutex;
    size_t points_per_pair;  // Pre-computed: width * height
    size_t num_pairs;

    // Constructor with pre-allocation
    FrameCloudAggregator(size_t num_pairs_, size_t points_per_pair_)
        : combined_cloud(std::make_shared<pcl::PointCloud<pcl::PointXYZ>>()),
          points_per_pair(points_per_pair_),
          num_pairs(num_pairs_) {
        // Pre-allocate full size to avoid reallocations
        combined_cloud->points.resize(num_pairs * points_per_pair);
        combined_cloud->width = num_pairs * points_per_pair;
        combined_cloud->height = 1;
        combined_cloud->is_dense = false;
    }

    FrameCloudAggregator() : combined_cloud(std::make_shared<pcl::PointCloud<pcl::PointXYZ>>()),
                             points_per_pair(0), num_pairs(0) {}
};

// Struct for tracking async timing per frame
struct FrameTimingData {
    std_msgs::msg::Header header;
    double preprocess_ms;
    double inference_wall_ms;  // Total wall time for inference phase
    std::map<std::string, double> rect_times;
    std::map<std::string, double> infer_times;
    std::map<std::string, double> cloud_times;
    std::map<std::string, double> viz_times;
    std::atomic<int> pairs_completed{0};
    std::mutex timing_mutex;

    FrameTimingData() = default;
};

class DepthEstimation : public rclcpp::Node {
public:
  explicit DepthEstimation(const rclcpp::NodeOptions &options);
  ~DepthEstimation();

private:
  // --- Initialization Functions ---
  void DeclareRosParameters();
  void InitializeRosParameters();
  void InitializeTransform();
  void InitializeModel();
  void InitializeCalibrationData();
  void InitializePublishers();
  void InitializeSubscribers();
  void InitializeServices();
  void InitializeLogging();
  void InitializePointcloudWorkers();
  void ShutdownPointcloudWorkers();

  // --- Core Logic & Callbacks ---
  void CompressedImageCallback(const sensor_msgs::msg::CompressedImage::SharedPtr msg);
  void RawImageCallback(const sensor_msgs::msg::Image::SharedPtr msg);

  // Core Processing Function
  void ProcessImage(const cv::Mat &concatenated_image,
                    const std_msgs::msg::Header &header,
                    const std::chrono::high_resolution_clock::time_point &t_start_total,
                    double preprocessing_ms);

  // Inference Helper
  cv::Mat RunInference(const cv::Mat &rectified_left,
                       const cv::Mat &rectified_right,
                       int session_idx);

  // Point Cloud Helper (optimized: single loop with combined transform)
  pcl::PointCloud<pcl::PointXYZ>::Ptr
  DisparityToPointCloud(const cv::Mat &disparity_map,
                        const cv::Mat &K_rect_left,
                        double baseline_meters,
                        double baseline_scale,
                        double disparity_offset,
                        const Eigen::Matrix4f &combined_transform);

  // Async pointcloud processing
  void PointcloudWorkerLoop();
  void EnqueueDisparity(DisparityPayload&& payload);
  void ProcessPointcloudAsync(DisparityPayload& payload);

  // Timing report helper
  void PrintFrameTimingReport(uint64_t frame_id);
  void LogFrameTiming(uint64_t frame_id);

  // --- Service Callback ---
  void GetCameraInfoCallback(
      const std::shared_ptr<depth_estimation_ros2::srv::GetCameraInfo::Request>
          request,
      std::shared_ptr<depth_estimation_ros2::srv::GetCameraInfo::Response>
          response);

  // --- Helper Functions ---
  std::string GetSystemArchitecture();
  std::string GetGpuName();
  std::string SanitizeString(std::string str);

  // --- ROS Parameters ---
  bool verbose_ = false;
  bool run_parallel_ = true;
  bool use_compressed_image_ = true;
  std::string input_image_topic_;
  std::string transform_config_path_;
  std::string onnx_model_path_;
  std::string model_type_;
  std::string execution_provider_;
  std::vector<std::string> stereo_pairs_;
  std::string calibration_base_path_;
  std::string calibration_resolution_;
  bool publish_combined_pointcloud_ = false;
  std::string combined_pointcloud_topic_;
  std::string pointcloud_frame_id_;
  bool filter_disparity_ = false;

  // NOTE: depth_correction parameters are loaded from calibration folder
  // and stored directly in StereoPairData, not as class members

  // Logging parameters
  bool enable_logging_ = false;
  std::string logging_directory_;
  std::ofstream timing_file_;
  std::mutex logging_mutex_;

  // --- ONNX Runtime Members ---
  Ort::Env ort_env_;
  Ort::SessionOptions session_options_;

  // Vector of sessions
  std::vector<std::shared_ptr<Ort::Session>> ort_sessions_;

  std::vector<std::string> input_node_names_;
  std::vector<std::string> output_node_names_;
  std::vector<std::vector<int64_t>> input_node_dims_;
  int model_full_height_ = 0;
  int model_full_width_ = 0;
  int model_half_height_ = 0;
  int model_half_width_ = 0;

  // --- Data and State ---
  bool is_initialized_ = false;
  int fisheye_image_width_ = 0;
  std::map<std::string, StereoPairData> calibration_data_;
  Eigen::Matrix4f transform_cam0_to_centroid_;

  // --- Async Pointcloud Processing ---
  std::queue<DisparityPayload> pointcloud_queue_;
  std::mutex queue_mutex_;
  std::condition_variable queue_cv_;
  std::atomic<bool> shutdown_requested_{false};
  std::vector<std::thread> pointcloud_workers_;
  int num_pointcloud_workers_ = 2;
  static constexpr size_t MAX_QUEUE_SIZE = 16;

  // Frame-based aggregation for combined pointcloud
  std::atomic<uint64_t> frame_counter_{0};
  std::map<uint64_t, std::shared_ptr<FrameCloudAggregator>> frame_aggregators_;
  std::mutex aggregator_mutex_;

  // --- Debug Grid (for verbose mode) ---
  std::mutex debug_grid_mutex_;
  std::map<uint64_t, std::vector<std::pair<std::string, cv::Mat>>> frame_debug_images_;

  // --- Frame Timing Tracking ---
  std::map<uint64_t, std::shared_ptr<FrameTimingData>> frame_timing_data_;
  std::mutex timing_data_mutex_;

  // --- ROS Communication ---
  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr compressed_image_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr raw_image_sub_;

  std::map<std::string,
           rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr>
      pointcloud_pubs_;

  // Single publisher for the stitched 2x2 grid
  rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr debug_grid_pub_;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
      combined_pointcloud_pub_;
  rclcpp::Service<depth_estimation_ros2::srv::GetCameraInfo>::SharedPtr
      camera_info_service_;
};

} // namespace depth_estimation

#endif // DEPTH_ESTIMATION_HPP_
