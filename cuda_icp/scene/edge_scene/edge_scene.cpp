#include "edge_scene.h"

using namespace cv;
using namespace std;

void Scene_edge::init_Scene_edge_cpu(cv::Mat dx_, cv::Mat dy_, std::vector<::Vec2f> &pcd_buffer,
                                     std::vector<::Vec2f>& normal_buffer, float max_dist_diff, float low_thresh, float high_thresh)
{
    assert(dx_.rows == dy_.rows && dx_.cols == dy_.cols);
    width = dx_.cols;
    height = dx_.rows;
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

    normal_buffer.clear();
    normal_buffer.resize(width * height);

    pcd_buffer.clear();
    pcd_buffer.resize(width * height, ::Vec2f(-1, -1)); // -1 indicate no edge around

    std::vector<::Vec2f> pcd_buffer_fixed = pcd_buffer;
    std::vector<::Vec2f> normal_buffer_fixed = normal_buffer;

#pragma omp parallel for collapse(2)
    for(int r=0; r<height; r++){
        for(int c=0; c<width; c++){
            if(edge.at<uchar>(r, c) > 0){  // get normals & pcds at edge only

                int w = dx.cols;
                int h = dx.rows;
                Point icontour = {c, r};

                vector<double> magNeighbour(9);
                songyuncen_subpixel::getMagNeighbourhood(dx, dy, icontour, w, h, magNeighbour);
                vector<double> a(9);
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

                normal_buffer_fixed[c + r*width] = {float(nx), float(-ny)};
                pcd_buffer_fixed[c +r*width] = {x, y};
            }
        }
    }
    // get pcd, dilute to neibor
    {
        // may padding to divid and parallel
        cv::Mat dist_buffer(height, width, CV_32FC1, FLT_MAX);
        int kernel_size = int(max_dist_diff+0.5f);
        for(int r=0+kernel_size; r<height - kernel_size; r++){
            for(int c=0+kernel_size; c<width - kernel_size; c++){

                if(edge.at<uchar>(r, c) > 0){
                    auto pcd = pcd_buffer_fixed[c + r*width];
                    for(int i=-kernel_size; i<=kernel_size; i++){
                        for(int j=-kernel_size; j<=kernel_size; j++){

                            float dist_sq = pow2(i) + pow2(j);
//                            float dist_sq = pow2(j-(pcd.x-c)) + pow2(i-(pcd.y-r));  // this is better?
                            // don't go too far
                            if(dist_sq > pow2(max_dist_diff)) continue;

                            int new_r = r + i;
                            int new_c = c + j;

                            // if closer
                            if(dist_sq < dist_buffer.at<float>(new_r, new_c)){
                                pcd_buffer[new_c + new_r*width] = pcd;
                                normal_buffer[new_c + new_r*width] = normal_buffer_fixed[c + r*width];
                                dist_buffer.at<float>(new_r, new_c) = dist_sq;
                            }
                        }
                    }
                }
            }
        }
    }

    pcd_ptr = pcd_buffer.data();
    normal_ptr = normal_buffer.data();
}


