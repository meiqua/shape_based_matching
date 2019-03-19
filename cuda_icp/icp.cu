#include "icp.h"
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>

namespace cuda_icp{
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


__global__ void transform_pcd_cuda(Vec2f* model_pcd_ptr, uint32_t model_pcd_size, Mat3x3f trans){
    uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= model_pcd_size) return;

    Vec2f& pcd = model_pcd_ptr[i];
    float new_x = trans[0][0]*pcd.x + trans[0][1]*pcd.y + trans[0][2];
    float new_y = trans[1][0]*pcd.x + trans[1][1]*pcd.y + trans[1][2];
    pcd.x = new_x;
    pcd.y = new_y;
}


template<class Scene>
RegistrationResult ICP2D_Point2Plane_cuda(device_vector_holder<Vec2f> &model_pcd, const Scene scene,
                                        const ICPConvergenceCriteria criteria){
    RegistrationResult result;
    RegistrationResult backup;

    thrust::host_vector<float> A_host(9, 0);
    thrust::host_vector<float> b_host(3, 0);

    const uint32_t threadsPerBlock = 256;
    const uint32_t numBlocks = (model_pcd.size() + threadsPerBlock - 1)/threadsPerBlock;

    for(uint32_t iter=0; iter<= criteria.max_iteration_; iter++){

        Vec11f Ab_tight = thrust::transform_reduce(thrust::cuda::par.on(cudaStreamPerThread),
                                        model_pcd.begin_thr(), model_pcd.end_thr(), thrust__pcd2Ab<Scene>(scene),
                                        Vec11f::Zero(), thrust__plus());

        cudaStreamSynchronize(cudaStreamPerThread);
        backup = result;

        float& count = Ab_tight[10];
        float& total_error = Ab_tight[9];
        if(count == 0) return result;  // avoid divid 0

        result.fitness_ = float(count) / model_pcd.size();
        result.inlier_rmse_ = std::sqrt(total_error / count);

        // last extra iter, just compute fitness & mse
        if(iter == criteria.max_iteration_) return result;

        if(std::abs(result.fitness_ - backup.fitness_) < criteria.relative_fitness_ &&
           std::abs(result.inlier_rmse_ - backup.inlier_rmse_) < criteria.relative_rmse_){
            return result;
        }

        for(int i=0; i<3; i++) b_host[i] = Ab_tight[6 + i];

        int shift = 0;
        for(int y=0; y<3; y++){
            for(int x=y; x<3; x++){
                A_host[x + y*3] = Ab_tight[shift];
                A_host[y + x*3] = Ab_tight[shift];
                shift++;
            }
        }

        Mat3x3f extrinsic = eigen_slover_333(A_host.data(), b_host.data());

        transform_pcd_cuda<<<numBlocks, threadsPerBlock>>>(model_pcd.data(), model_pcd.size(), extrinsic);
        cudaStreamSynchronize(cudaStreamPerThread);

        result.transformation_ = extrinsic * result.transformation_;
    }

    // never arrive here
    return result;
}

template RegistrationResult ICP2D_Point2Plane_cuda(device_vector_holder<Vec2f> &model_pcd, const Scene_edge scene,
                                        const ICPConvergenceCriteria criteria);
}



