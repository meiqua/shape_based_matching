cmake_minimum_required(VERSION 2.8)
set (CMAKE_CXX_STANDARD 14)
project(line2Dup_pybind)

SET(CMAKE_BUILD_TYPE "Release")
#SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

set(CMAKE_PREFIX_PATH /opt/ros/kinetic)
find_package(OpenCV 3 REQUIRED)
#message(STATUS "dir='${OpenCV_DIR}'")

find_package(PythonLibs 3 REQUIRED)
find_package(pybind11 REQUIRED)

include(Cmake/FindNumPy.cmake)

include_directories(${OpenCV_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS}
    ${PYTHON_NUMPY_INCLUDE_DIR})

set(SOURCE_PYBIND
    pybind11.cpp
    np2mat/ndarray_converter.cpp)

set(SOURCE
    line2Dup.cpp)

pybind11_add_module(${PROJECT_NAME} ${SOURCE} ${SOURCE_PYBIND})
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})

add_executable(${PROJECT_NAME}_executable ${SOURCE} test.cpp)
target_link_libraries(${PROJECT_NAME}_executable ${OpenCV_LIBS})

