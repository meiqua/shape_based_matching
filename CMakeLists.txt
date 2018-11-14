cmake_minimum_required(VERSION 2.8)
set (CMAKE_CXX_STANDARD 14)
project(line2Dup_pybind)

SET(CMAKE_BUILD_TYPE "Release")
SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -fopenmp -O0 -Wall -g2 -ggdb -Wno-sign-compare")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -fopenmp -O3 -Wall -Wno-sign-compare")

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} /opt/ros/kinetic)
find_package(OpenCV 3 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# include MIPP headers
include_directories (${INCLUDE_DIRECTORIES} "${CMAKE_CURRENT_SOURCE_DIR}/MIPP/")

set(SOURCE
    line2Dup.cpp)

add_executable(${PROJECT_NAME}_executable ${SOURCE} test.cpp)
target_link_libraries(${PROJECT_NAME}_executable ${OpenCV_LIBS})

