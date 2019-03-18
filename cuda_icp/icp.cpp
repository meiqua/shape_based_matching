#include "icp.h"
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/Geometry>

namespace cuda_icp{

Eigen::Matrix3d TransformVector3dToMatrix3d(const Eigen::Matrix<double, 3, 1> &input) {
    Eigen::Matrix3d output =
            (Eigen::AngleAxisd(input(0), Eigen::Vector3d::UnitZ()))
                    .matrix();
    output.block<2, 1>(0, 2) = input.block<2, 1>(1, 0);
    return output;
}

Mat3x3f eigen_to_custom(const Eigen::Matrix3f& extrinsic){
    Mat3x3f result;
    for(uint32_t i=0; i<3; i++){
        for(uint32_t j=0; j<3; j++){
            result[i][j] = extrinsic(i, j);
        }
    }
    return result;
}

Mat3x3f eigen_slover_333(float *A, float *b)
{
    Eigen::Matrix<float, 3, 3> A_eigen(A);
    Eigen::Matrix<float, 3, 1> b_eigen(b);
    const Eigen::Matrix<double, 3, 1> update = A_eigen.cast<double>().ldlt().solve(b_eigen.cast<double>());
    Eigen::Matrix3d extrinsic = TransformVector3dToMatrix3d(update);
    return eigen_to_custom(extrinsic.cast<float>());
}

void transform_pcd(std::vector<Vec2f>& model_pcd, Mat3x3f& trans){

#pragma omp parallel for
    for(uint32_t i=0; i < model_pcd.size(); i++){
        Vec2f& pcd = model_pcd[i];
        float new_x = trans[0][0]*pcd.x + trans[0][1]*pcd.y + trans[0][2];
        float new_y = trans[1][0]*pcd.x + trans[1][1]*pcd.y + trans[1][2];
        pcd.x = new_x;
        pcd.y = new_y;
    }
}

template<class Scene>
RegistrationResult ICP2D_Point2Plane_cpu(std::vector<Vec2f> &model_pcd, const Scene scene,
                                       const ICPConvergenceCriteria criteria)
{
    RegistrationResult result;
    RegistrationResult backup;

    std::vector<float> A_host(9, 0);
    std::vector<float> b_host(3, 0);
    thrust__pcd2Ab<Scene> trasnformer(scene);

    // use one extra turn
    for(uint32_t iter=0; iter<=criteria.max_iteration_; iter++){

        Vec11f reducer;

#pragma omp declare reduction( + : Vec11f : omp_out += omp_in) \
                       initializer (omp_priv = Vec11f::Zero())

#pragma omp parallel for reduction(+: reducer)
        for(size_t pcd_iter=0; pcd_iter<model_pcd.size(); pcd_iter++){
            Vec11f result = trasnformer(model_pcd[pcd_iter]);
            reducer += result;
        }

        Vec11f& Ab_tight = reducer;

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

        transform_pcd(model_pcd, extrinsic);
        result.transformation_ = extrinsic * result.transformation_;
    }

    // never arrive here
    return result;
}

template RegistrationResult ICP2D_Point2Plane_cpu(std::vector<Vec2f> &model_pcd, const Scene_edge scene,
const ICPConvergenceCriteria criteria);
}





