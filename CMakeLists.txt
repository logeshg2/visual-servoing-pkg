cmake_minimum_required(VERSION 3.8)
project(visual_servoing_pkg)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(std_msgs REQUIRED)

set(action_files
)
set(msg_files
  "msg/ArucoCorner.msg"
  "msg/MatchedPoints.msg"
)
set(srv_files
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${action_files}
  ${msg_files}
  ${srv_files}
  DEPENDENCIES std_msgs
)
ament_export_dependencies(rosidl_default_runtime)

# Install Python executables
install(PROGRAMS
  scripts/keypoints_eg.py
  scripts/black_circle.py
  scripts/aruco_node.py
  scripts/aruco_board_node.py
  scripts/superglue_n_points.py
  visual_servoing_pkg/ibvs_aruco_corner.py
  visual_servoing_pkg/pbvs_aruco_pose.py
  visual_servoing_pkg/ibvs_n_points.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})

ament_package()
