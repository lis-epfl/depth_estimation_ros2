#include "depth_estimation_ros2/depth_estimation.hpp"
#include "rclcpp/rclcpp.hpp"

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::executors::SingleThreadedExecutor exec;
  rclcpp::NodeOptions options;
  auto node = std::make_shared<depth_estimation::DepthEstimation>(options);
  exec.add_node(node);
  exec.spin();
  rclcpp::shutdown();
  return 0;
}
