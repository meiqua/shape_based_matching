#include "edge_scene.h"

void Scene_edge::init_Scene_edge_cuda(cv::Mat img, device_vector_holder<Vec2f> &pcd_buffer,
                                      device_vector_holder<Vec2f>& normal_buffer, float max_dist_diff)
{
    std::vector<Vec2f> pcd_buffer_host, normal_buffer_host;

    init_Scene_edge_cpu(img, pcd_buffer_host, normal_buffer_host, max_dist_diff);

    pcd_buffer.__malloc(pcd_buffer_host.size());
    thrust::copy(pcd_buffer_host.begin(), pcd_buffer_host.end(), pcd_buffer.begin_thr());

    normal_buffer.__malloc(normal_buffer_host.size());
    thrust::copy(normal_buffer_host.begin(), normal_buffer_host.end(), normal_buffer.begin_thr());

    pcd_ptr = pcd_buffer.data();
    normal_ptr = normal_buffer.data();
}
