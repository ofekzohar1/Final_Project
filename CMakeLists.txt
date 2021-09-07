cmake_minimum_required(VERSION 3.20)
project(Final_Project C)

set(CMAKE_C_STANDARD 90)
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

add_executable(Final_Project spkmeans.c spkmeansmodule.c)
target_link_libraries(Final_Project ${PYTHON_LIBRARIES})
