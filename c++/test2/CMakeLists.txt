cmake_minimum_required(VERSION 2.8)
project(test2)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-O3")

# 添加Eigen头文件
include_directories("/usr/include/eigen3")
add_executable(test2 test2.cpp)