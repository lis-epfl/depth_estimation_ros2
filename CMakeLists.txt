cmake_minimum_required(VERSION 3.8)
project(depth_estimation_ros2)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -std=c++17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(Eigen3 REQUIRED)

find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
find_package(yaml-cpp REQUIRED)

# Need CUDA only for getting GPU Name (driver API), not OpenCV-CUDA modules
find_package(CUDA REQUIRED)

find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h)
find_library(ONNXRUNTIME_LIBRARY onnxruntime)
if (NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR "ONNX Runtime not found.")
else()
    message(STATUS "Found ONNX Runtime: ${ONNXRUNTIME_LIBRARY}")
endif()

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/GetCameraInfo.srv"
  "msg/CameraInfo.msg"
  DEPENDENCIES geometry_msgs
)

rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")

add_executable(depth_estimation_node
  src/depth_estimation_node.cpp
  src/depth_estimation.cpp
)

ament_target_dependencies(depth_estimation_node
  rclcpp
  rclcpp_components
  sensor_msgs
  cv_bridge
  pcl_conversions
  geometry_msgs
  tf2_eigen
)

target_include_directories(depth_estimation_node PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(depth_estimation_node
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
  yaml-cpp
  ${ONNXRUNTIME_LIBRARY}
  ${CUDA_LIBRARIES}
  ${cpp_typesupport_target}
)

add_dependencies(depth_estimation_node ${PROJECT_NAME})

install(TARGETS
  depth_estimation_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

# Install the script into the package's lib directory
install(PROGRAMS
  scripts/download_models.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_export_include_directories(include)
ament_export_dependencies(rosidl_default_runtime)
ament_package()
