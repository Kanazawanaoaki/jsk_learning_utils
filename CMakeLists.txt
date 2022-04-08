cmake_minimum_required(VERSION 3.0.2)
project(jsk_learning_utils)

find_package(catkin REQUIRED)
include(cmake/setup_python_as_submodule.cmake)
setup_python_as_submodule(pythonlib)
catkin_package(
)

include_directories(
)
