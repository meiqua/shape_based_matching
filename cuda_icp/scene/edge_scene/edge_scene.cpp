#include "edge_scene.h"

void Scene_edge::init_Scene_edge_cpu(cv::Mat img, std::vector<Vec2f> &pcd_buffer,
                                     std::vector<Vec2f>& normal_buffer, float max_dist_diff)
{
    width = img.cols;
    height = img.rows;
    this->max_dist_diff = max_dist_diff;

    cv::Mat gray;
    if(img.channels() > 1){
        cv::cvtColor(img, gray, CV_BGR2GRAY);
    }else{
        gray = img;
    }

    cv::Mat smoothed = gray;
    static const int KERNEL_SIZE = 5;
    cv::GaussianBlur(gray, smoothed, cv::Size(KERNEL_SIZE, KERNEL_SIZE), 0, 0, cv::BORDER_REPLICATE);

    cv::Mat edge;
    cv::Canny(smoothed, edge, 30, 60);

//    cv::imshow("edge", edge);
//    cv::waitKey(0);

    // get normals
    { // edge direction; may reuse shape_based_matching's to save time
        // calculate from canny edge is faster?
        cv::Mat sobel_dx, sobel_dy, magnitude, sobel_ag;
        cv::Sobel(smoothed, sobel_dx, CV_32F, 1, 0, 3, 1.0, 0.0, cv::BORDER_REPLICATE);
        cv::Sobel(smoothed, sobel_dy, CV_32F, 0, 1, 3, 1.0, 0.0, cv::BORDER_REPLICATE);
        phase(sobel_dx, sobel_dy, sobel_ag, false);

        normal_buffer.clear();
        normal_buffer.resize(img.rows * img.cols);

        for(int r=0; r<img.rows; r++){
            for(int c=0; c<img.cols; c++){
                if(edge.at<uchar>(r, c) > 0){  // get normals at edge only

                    float theta = sobel_ag.at<float>(r, c);
                    normal_buffer[c + r*img.cols] = {
                        std::cos(theta),  // x is pointing right
                        -std::sin(theta)  // y is pointing down
                    };
                }
            }
        }
    }


    // get pcd
    {
        pcd_buffer.clear();
        pcd_buffer.resize(img.rows * img.cols, Vec2f(-1, -1)); // -1 indicate no edge around

        // may padding to divid and parallel
        cv::Mat dist_buffer(img.size(), CV_32FC1, FLT_MAX);
        int kernel_size = int(max_dist_diff+0.5f);
        for(int r=0+kernel_size; r<img.rows - kernel_size; r++){
            for(int c=0+kernel_size; c<img.cols - kernel_size; c++){

                if(edge.at<uchar>(r, c) > 0){

                    for(int i=-kernel_size; i<=kernel_size; i++){
                        for(int j=-kernel_size; j<=kernel_size; j++){

                            float dist_sq = i*i + j*j;

                            // don't go too far
                            if(dist_sq > max_dist_diff*max_dist_diff) continue;

                            int new_r = r + i;
                            int new_c = c + j;

                            // if closer
                            if(dist_sq < dist_buffer.at<float>(new_r, new_c)){
                                pcd_buffer[new_c + new_r*img.cols] = {
                                    float(c),
                                    float(r)
                                };
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


