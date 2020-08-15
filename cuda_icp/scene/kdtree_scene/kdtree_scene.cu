#include "edge_scene.h"

void Scene_edge::init_Scene_edge_cuda(cv::Mat dx, cv::Mat dy, KDTree_cuda& kdtree, float max_dist_diff,
                                      float low_thresh, float high_thresh)
{
    KDTree_cpu cpu_tree;
    init_Scene_edge_cpu(dx, dy, cpu_tree, max_dist_diff, low_thresh, high_thresh);

    kdtree.pcd_buffer.__malloc(cpu_tree.pcd_buffer.size());
    thrust::copy(cpu_tree.pcd_buffer.begin(), cpu_tree.pcd_buffer.end(), kdtree.pcd_buffer.begin_thr());

    kdtree.normal_buffer.__malloc(cpu_tree.normal_buffer.size());
    thrust::copy(cpu_tree.normal_buffer.begin(), cpu_tree.normal_buffer.end(), kdtree.normal_buffer.begin_thr());

    kdtree.nodes.__malloc(cpu_tree.nodes.size());
    thrust::copy(cpu_tree.nodes.begin(), cpu_tree.nodes.end(), kdtree.nodes.begin_thr());

    pcd_ptr = kdtree.pcd_buffer.data();
    normal_ptr = kdtree.normal_buffer.data();
    node_ptr = kdtree.nodes.data();
}
