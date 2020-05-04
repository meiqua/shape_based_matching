#include "line2Dup.h"
#include <memory>
#include <iostream>
#include <assert.h>
#include <chrono>
#include <map>
#include <stdlib.h>
using namespace std;
using namespace cv;

#include "fusion.h"

static std::string prefix = "/home/rfjiang/shape_based_matching/test/";
void gauss_test()
{
    // only support gray img now
    Mat test_img = imread(prefix + "case1/test.png", cv::IMREAD_GRAYSCALE);
    assert(!test_img.empty() && "check your img path");

    int padding = 1500;
    cv::Mat padded_img = cv::Mat(test_img.rows + 2 * padding,
                                 test_img.cols + 2 * padding, test_img.type(), cv::Scalar::all(0));
    test_img.copyTo(padded_img(Rect(padding, padding, test_img.cols, test_img.rows)));

    int stride = 16;
    int n = padded_img.rows / stride;
    int m = padded_img.cols / stride;
    Rect roi(0, 0, stride * m, stride * n);
    Mat img = padded_img(roi).clone();
    assert(img.isContinuous());

    std::cout << "test img size: " << img.rows * img.cols << std::endl
              << std::endl;

    Timer timer;
    double opencv_time = 0;
    static const int KERNEL_SIZE = 5;
    Mat smoothed;
    GaussianBlur(img, smoothed, Size(KERNEL_SIZE, KERNEL_SIZE), 0, 0, BORDER_CONSTANT);
    opencv_time += timer.out("GaussianBlur");

    std::cout << "opencv total time: " << opencv_time << std::endl;

    std::vector<cv::Mat> in_v;
    in_v.push_back(img);
    std::vector<cv::Mat> out_v;
    Mat mag_fusion(img.size(), CV_16S, cv::Scalar(0));
    out_v.push_back(mag_fusion);

    simple_fusion::ProcessManager manager(32, 256);
    manager.nodes_.clear();
    manager.nodes_.push_back(std::make_shared<simple_fusion::Gauss1x5Node_8U_32S_4bit_larger>());
    manager.nodes_.push_back(std::make_shared<simple_fusion::Gauss5x1Node_32S_16S_4bit_smaller>());
    manager.arrange(img.rows, img.cols);

    timer.reset();
    manager.process(in_v, out_v);
    timer.out("fusion mag");

    mag_fusion.convertTo(mag_fusion, CV_8U);
    Mat mag_diff = cv::abs(smoothed - mag_fusion);

    imshow("img", img);
    imshow("diff", mag_diff > 1);  // we may have 1 diff due to quantization
    waitKey(0);

    std::cout << "test end" << std::endl;
}

void mag_test()
{
    // only support gray img now
    Mat test_img = imread(prefix + "case1/test.png", cv::IMREAD_GRAYSCALE);
    assert(!test_img.empty() && "check your img path");

    int padding = 500;
    cv::Mat padded_img = cv::Mat(test_img.rows + 2 * padding,
                                 test_img.cols + 2 * padding, test_img.type(), cv::Scalar::all(0));
    test_img.copyTo(padded_img(Rect(padding, padding, test_img.cols, test_img.rows)));

    int stride = 16;
    int n = padded_img.rows / stride;
    int m = padded_img.cols / stride;
    Rect roi(0, 0, stride * m, stride * n);
    Mat img = padded_img(roi).clone();
    assert(img.isContinuous());

    std::cout << "test img size: " << img.rows * img.cols << std::endl
              << std::endl;

    Timer timer;
    double opencv_time = 0;
    static const int KERNEL_SIZE = 5;
    Mat smoothed;
    GaussianBlur(img, smoothed, Size(KERNEL_SIZE, KERNEL_SIZE), 0, 0, BORDER_CONSTANT);
    opencv_time += timer.out("GaussianBlur");

    Mat sobel_dx, sobel_dy, sobel_ag;
    Sobel(smoothed, sobel_dx, CV_32F, 1, 0, 3, 1.0, 0.0, BORDER_CONSTANT);
    opencv_time += timer.out("sobel_dx");

    Sobel(smoothed, sobel_dy, CV_32F, 0, 1, 3, 1.0, 0.0, BORDER_CONSTANT);
    opencv_time += timer.out("sobel_dy");

    Mat magnitude = sobel_dx.mul(sobel_dx) + sobel_dy.mul(sobel_dy);
    opencv_time += timer.out("magnitude");

    std::cout << "opencv mag total time: " << opencv_time << std::endl;

    std::vector<cv::Mat> in_v;
    in_v.push_back(img);
    std::vector<cv::Mat> out_v;
    Mat mag_fusion(img.size(), CV_32S, cv::Scalar(0));
    out_v.push_back(mag_fusion);

    simple_fusion::ProcessManager manager;
    manager.nodes_.clear();
    manager.nodes_.push_back(std::make_shared<simple_fusion::Gauss1x5Node_8U_32S_4bit_larger>());
    manager.nodes_.push_back(std::make_shared<simple_fusion::Gauss5x1Node_32S_16S_4bit_smaller>());
    manager.nodes_.push_back(std::make_shared<simple_fusion::Sobel1x3SxxSyxNode_16S_16S>());
    manager.nodes_.push_back(std::make_shared<simple_fusion::Sobel3x1SxySyyNode_16S_16S>());
    manager.nodes_.push_back(std::make_shared<simple_fusion::MagSqure1x1Node_16S_32S>());
    manager.arrange(img.rows, img.cols);

    timer.reset();
    manager.process(in_v, out_v);
    timer.out("fusion mag");

    mag_fusion.convertTo(mag_fusion, CV_32F);
    Mat mag_diff = cv::abs(magnitude - mag_fusion);

    imshow("img", img);
    imshow("diff", mag_diff);
    waitKey(0);

    std::cout << "test end" << std::endl;
}


int main()
{
    mag_test();
    return 0;
}
