cmake_minimum_required(VERSION 3.8)
project(visual_servoing_pkg)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)

# Install Python executables
install(PROGRAMS
  scripts/keypoints_eg.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
