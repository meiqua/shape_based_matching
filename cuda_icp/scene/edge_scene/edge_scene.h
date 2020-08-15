#pragma once

#include "../common.h"

// frame of scene edge
// o -------> x
// |
// |
// |
// V
// y


// just implement query func
struct Scene_edge{
    size_t width = 640, height = 480;
    float max_dist_diff = 4.0f; // pixels
    Vec2f* pcd_ptr;  // pointer can unify cpu & cuda version
    Vec2f* normal_ptr;  // layout: 1d, width*height length, array of Vec2f

    // buffer provided by user, this class only holds pointers,
    // becuase we will pass them to device.
    void init_Scene_edge_cpu(cv::Mat dx, cv::Mat dy, std::vector<Vec2f>& pcd_buffer,
                             std::vector<Vec2f>& normal_buffer, float max_dist_diff = 4.0f,
                             float low_thresh = 30, float high_thresh = 60);

#ifdef CUDA_ON
    void init_Scene_edge_cuda(cv::Mat dx, cv::Mat dy, device_vector_holder<Vec2f>& pcd_buffer,
                              device_vector_holder<Vec2f>& normal_buffer, float max_dist_diff = 4.0f,
                              float low_thresh = 30, float high_thresh = 60);
#endif

    __device__ __host__
    void query(const Vec2f& src_pcd, Vec2f& dst_pcd, Vec2f& dst_normal, bool& valid) const {

        size_t x,y;
        x = size_t(src_pcd.x + 0.5f);
        y = size_t(src_pcd.y + 0.5f);

        if(x >= width || y >= height){
            valid = false;
            return;
        }

        size_t idx = x + y * width;
        if(pcd_ptr[idx].x >= 0){

            dst_pcd = pcd_ptr[idx];

            dst_normal = normal_ptr[idx];

            valid = true;

        }else valid = false;

        return;
    }
};
