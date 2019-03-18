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
    void init_Scene_edge_cpu(cv::Mat img, std::vector<Vec2f>& pcd_buffer,
                             std::vector<Vec2f>& normal_buffer, float max_dist_diff = 4.0f);

#ifdef CUDA_ON
    void init_Scene_edge_cuda(cv::Mat img, device_vector_holder<Vec2f>& pcd_buffer,
                              device_vector_holder<Vec2f>& normal_buffer, float max_dist_diff = 4.0f);
#endif

    __device__ __host__
    void query(const Vec2f& src_pcd, Vec2f& dst_pcd, Vec2f& dst_normal, bool& valid) const {
        size_t idx = size_t(src_pcd.x + 0.5f) + size_t(src_pcd.y + 0.5f) * width;
        if(pcd_ptr[idx].x >= 0){

            dst_pcd = pcd_ptr[idx];

            idx = size_t(dst_pcd.x) + size_t(dst_pcd.y) * width;
            dst_normal = normal_ptr[idx];

            valid = true;

        }else valid = false;

        return;
    }
};
