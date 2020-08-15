#include "kdtree_scene.h"

void Scene_kdtree::init_Scene_kdtree_cpu(cv::Mat dx_, cv::Mat dy_, KDTree_cpu& kdtree,
                                    float max_dist_diff, float low_thresh, float high_thresh)
{
    assert(dx_.rows == dy_.rows && dx_.cols == dy_.cols);
    int width = dx_.cols;
    int height = dx_.rows;
    this->max_dist_diff = max_dist_diff;

    cv::Mat dx, dy;
    if(dx_.type() != CV_16S)
        dx_.convertTo(dx, CV_16S);
    else
        dx = dx_;

    if(dy_.type() != CV_16S)
        dy_.convertTo(dy, CV_16S);
    else
        dy = dy_;

    cv::Mat edge;
    cv::Canny(dx, dy, edge, low_thresh, high_thresh);

    kdtree.normal_buffer.clear();
    kdtree.pcd_buffer.clear();
    for(int r=0; r<height; r++){
        for(int c=0; c<width; c++){
            if(edge.at<uchar>(r, c) > 0){  // get normals & pcds at edge only

                int w = dx.cols;
                int h = dx.rows;
                cv::Point icontour = {c, r};

                std::vector<double> magNeighbour(9);
                songyuncen_subpixel::getMagNeighbourhood(dx, dy, icontour, w, h, magNeighbour);
                std::vector<double> a(9);
                songyuncen_subpixel::get2ndFacetModelIn3x3(magNeighbour, a);

                // Hessian eigen vector
                double eigvec[2][2], eigval[2];
                songyuncen_subpixel::eigenvals(a, eigval, eigvec);
                double t = 0.0;
                double ny = eigvec[0][0];
                double nx = eigvec[0][1];
                if (eigval[0] < 0.0)
                {
                    double rx = a[1], ry = a[2], rxy = a[4], rxx = a[3] * 2.0, ryy = a[5] * 2.0;
                    t = -(rx * nx + ry * ny) / (rxx * nx * nx + 2.0 * rxy * nx * ny + ryy * ny * ny);
                }
                double px = nx * t;
                double py = ny * t;
                float x = (float)icontour.x;
                float y = (float)icontour.y;
                if (fabs(px) <= 0.5 && fabs(py) <= 0.5)
                {
                    x += (float)px;
                    y += (float)py;
                }

                kdtree.normal_buffer.push_back({float(nx), float(-ny)});
                kdtree.pcd_buffer.push_back({x, y});
            }
        }
    }
    kdtree.build_tree();

    pcd_ptr = kdtree.pcd_buffer.data();
    normal_ptr = kdtree.normal_buffer.data();
    node_ptr = kdtree.nodes.data();
}

// non-recursion non-stack building
// make it easy to transform to cuda implementation
// the remained hard part is that idx of nodes are dependent
// use cuda atomicAdd(fetch and add) looks fine
// not enough, may need Dynamic Parallelism in cuda
void KDTree_cpu::build_tree(int max_num_pcd_in_leaf)
{
    assert(pcd_buffer.size() > 0 && pcd_buffer.size() == normal_buffer.size()
           && "no pcd yet, or pcd size != normal size");

    std::vector<int> index;
    index.resize(pcd_buffer.size());
    std::iota(std::begin(index), std::end(index), 0); // Fill with 0, 1, 2, ...

    std::vector<int> index_buffer(index.size());

    //root
    nodes.resize(1);
    nodes[0].left = 0;
    nodes[0].right = index.size();

    bool exist_new_nodes = false;
    size_t num_nodes_now = 1;
    size_t num_nodes_last_last_turn = 0;
    size_t num_nodes_now_last_turn = 0;
    bool stop = false;

    while (!stop) { // when we have new nodes to split, go

        nodes.resize(num_nodes_now*2+1); // we may increase now + 1 in 1 turn

        exist_new_nodes = false; // reset
        num_nodes_last_last_turn = num_nodes_now_last_turn;
        num_nodes_now_last_turn = num_nodes_now; // for iter, avoid reaching new node in 1 turn

        // if you want to implement cuda version, paralleling this loop looks fine
        for(size_t node_iter = num_nodes_last_last_turn; node_iter < num_nodes_now_last_turn; node_iter++){

            // not a leaf
            if(nodes[node_iter].right - nodes[node_iter].left > max_num_pcd_in_leaf){

                // split start <----------------------
                // get bbox
                float x_min = FLT_MAX; float x_max = -FLT_MAX;  // fxxk, FLT_MIN is 0
                float y_min = FLT_MAX; float y_max = -FLT_MAX;
                for(int idx_iter = nodes[node_iter].left; idx_iter < nodes[node_iter].right; idx_iter++){
                    const auto& p = pcd_buffer[index[idx_iter]];
                    if(p.x > x_max) x_max = p.x;
                    if(p.x < x_min) x_min = p.x;
                    if(p.y > y_max) y_max = p.y;
                    if(p.y < y_min) y_min = p.y;
                }

                // select split dim & value
                int split_dim = 0;
                float split_val = 0;
                float span_xyz[2], split_v_xyz[2];
                float max_span = -FLT_MAX;
                span_xyz[0] = x_max - x_min; split_v_xyz[0] = (x_min + x_max)/2;
                span_xyz[1] = y_max - y_min; split_v_xyz[1] = (y_min + y_max)/2;
                for(int span_iter=0; span_iter<2; span_iter++){
                    if(span_xyz[span_iter] > max_span){
                        max_span = span_xyz[span_iter];
                        split_dim = span_iter;
                        split_val = split_v_xyz[span_iter];
                    }
                }

                // reorder index
                int left_iter = nodes[node_iter].left;
                int right_iter = nodes[node_iter].right - 1;
                float split_low = -FLT_MAX;
                float split_high = FLT_MAX;

                bool lr_switch = true;
                for(int idx_iter = nodes[node_iter].left; idx_iter<nodes[node_iter].right; idx_iter++){
                    float p = pcd_buffer[index[idx_iter]][split_dim];

                    if(p == split_val) lr_switch = !lr_switch;

                    if(p < split_val || (p==split_val && lr_switch)){
                        index_buffer[left_iter] = index[idx_iter];
                        left_iter ++;
                        if(p > split_low) split_low = p;
                    }else{
                        index_buffer[right_iter] = index[idx_iter];
                        right_iter --;
                        if(p < split_high) split_high = p;
                    }
                }
                assert(left_iter == right_iter + 1 && "left & right should meet");
                split_val = (split_low + split_high)/2; // reset split_val to middle

                for(int idx_iter = nodes[node_iter].left; idx_iter<nodes[node_iter].right; idx_iter++){
                    index[idx_iter] = index_buffer[idx_iter];
                }
                // split success <----------------------

                // update parent
                nodes[node_iter].child1 = num_nodes_now;
                nodes[node_iter].child2 = num_nodes_now + 1;
                nodes[node_iter].split_v = split_val;
                nodes[node_iter].split_dim = split_dim;
                nodes[node_iter].bbox[0] = x_min;  nodes[node_iter].bbox[1] = x_max;
                nodes[node_iter].bbox[2] = y_min;  nodes[node_iter].bbox[3] = y_max;

                // update child
                nodes[num_nodes_now].left = nodes[node_iter].left;
                nodes[num_nodes_now].right = left_iter;
                nodes[num_nodes_now].parent = node_iter;

                nodes[num_nodes_now + 1].left = left_iter;
                nodes[num_nodes_now + 1].right = nodes[node_iter].right;
                nodes[num_nodes_now + 1].parent = node_iter;

                num_nodes_now += 2;
                if(!exist_new_nodes) exist_new_nodes = true;
            }
        }

        if(!exist_new_nodes){
            stop = true;
        }
    }

    // we may give nodes more memory while spliting
    nodes.resize(num_nodes_now);

    // reorder pcd normal according to index, so avoid using index when query
    std::vector<Vec2f> v2f_buffer(pcd_buffer.size());
    for(size_t i=0; i<pcd_buffer.size(); i++){
        v2f_buffer[i] = pcd_buffer[index[i]];
    }
    pcd_buffer = v2f_buffer;

    for(size_t i=0; i<normal_buffer.size(); i++){
        v2f_buffer[i] = normal_buffer[index[i]];
    }
    normal_buffer = v2f_buffer;
}
