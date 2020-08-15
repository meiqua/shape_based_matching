#pragma once

#include "../common.h"

// frame of scene edge
// o -------> x
// |
// |
// |
// V
// y

struct Node_kdtree{
    // tree info
    int parent = -1;
    int child1 = -1;
    int child2 = -1;

    // non-leaf info
    float split_v;
    float bbox[4];  // x_min x_max y...
    int split_dim;

    // leaf info
    int left;
    int right;

    __device__ __host__
    bool isleaf(){
        if(child1 < 0 || child2 < 0) return true;
        return false;
    }
};

class KDTree_cpu{
public:
    std::vector<Vec2f> pcd_buffer;
    std::vector<Vec2f> normal_buffer;
    std::vector<Node_kdtree> nodes;

    void build_tree(int max_num_pcd_in_leaf = 10);
};


#ifdef CUDA_ON
class KDTree_cuda{
public:
    device_vector_holder<Vec2f> pcd_buffer;
    device_vector_holder<Vec2f> normal_buffer;
    device_vector_holder<Node_kdtree> nodes;
};
#endif

// just implement query func
struct Scene_kdtree{
    float max_dist_diff = 4.0f; // pixels
    Vec2f* pcd_ptr;  // pointer can unify cpu & cuda version
    Vec2f* normal_ptr;
    Node_kdtree* node_ptr;
    // buffer provided by user, this class only holds pointers,
    // becuase we will pass them to device.
    void init_Scene_kdtree_cpu(cv::Mat dx, cv::Mat dy, KDTree_cpu& kdtree, float max_dist_diff = 4.0f,
                             float low_thresh = 30, float high_thresh = 60);

#ifdef CUDA_ON
    void init_Scene_kdtree_cuda(cv::Mat dx, cv::Mat dy, KDTree_cuda& kdtree, float max_dist_diff = 4.0f,
                              float low_thresh = 30, float high_thresh = 60);
#endif

   __device__ __host__
    void query(const Vec2f& src_pcd, Vec2f& dst_pcd, Vec2f& dst_normal, bool& valid) const {
        bool backtrack=false;
        int lastNode=-1;
        int current=0;
        int result_idx = 0;
        float cloest_dist_sq = FLT_MAX;

        Node_kdtree node_cur;

        assert(node_ptr[0].parent == -1);
        while (current >= 0) { // parent of root is -1
             node_cur = node_ptr[current];

             float diff = 0;
             if (node_cur.split_dim == 0) diff = src_pcd.x - node_cur.split_v;
             if (node_cur.split_dim == 1) diff = src_pcd.y - node_cur.split_v;

             int best_child = node_cur.child1;
             int the_other = node_cur.child1;
             if(diff < 0) the_other = node_cur.child2;
             else best_child = node_cur.child2;

             if(!backtrack){
                 if(node_cur.isleaf()){
                     for(int i=node_cur.left; i<node_cur.right; i++){
                         float cur_dist_sq =
                                 pow2(src_pcd.x - pcd_ptr[i].x) +
                                 pow2(src_pcd.y - pcd_ptr[i].y);
                         if( cur_dist_sq < cloest_dist_sq ){
                             cloest_dist_sq = cur_dist_sq;
                             result_idx = i;
                         }
                     }
                     backtrack = true;
                     lastNode = current;
                     current = node_cur.parent;
                 }else{
                     lastNode = current;
                     current = best_child; // go down
                 }
             }else{
                 float min_possible_dist_sq = 0;

                 if(src_pcd.x < node_cur.bbox[0]) min_possible_dist_sq += pow2(node_cur.bbox[0] - src_pcd.x);
            else if(src_pcd.x > node_cur.bbox[1]) min_possible_dist_sq += pow2(node_cur.bbox[1] - src_pcd.x);
                 if(src_pcd.y < node_cur.bbox[2]) min_possible_dist_sq += pow2(node_cur.bbox[2] - src_pcd.y);
            else if(src_pcd.y > node_cur.bbox[3]) min_possible_dist_sq += pow2(node_cur.bbox[3] - src_pcd.y);

                 //  the far node was NOT the last node (== not visited yet),
                 //  AND there could be a closer point in it
                 if((lastNode == best_child) && (min_possible_dist_sq <= cloest_dist_sq)){
                     lastNode=current;
                     current=the_other;
                     backtrack=false;
                 }
                 else{
                     lastNode=current;
                     current=node_cur.parent;
                 }
             }
        }

        if(cloest_dist_sq < pow2(max_dist_diff)){
            valid = true;
            dst_pcd = pcd_ptr[result_idx];
            dst_normal = normal_ptr[result_idx];
            return;
        }else{
            valid = false;
            return;
        }
    }
};
