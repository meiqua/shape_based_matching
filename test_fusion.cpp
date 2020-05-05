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

    simple_fusion::ProcessManager manager;
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

void sobel_mag_test()
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
    imshow("img", img);
    std::cout << "test img size: " << img.rows * img.cols << std::endl
              << std::endl;

    Timer timer;
    double opencv_time = 0;

    Mat img16;
    img.convertTo(img16, CV_16S);

    Mat sobel_dx, sobel_dy, sobel_ag;
    Sobel(img16, sobel_dx, CV_16S, 1, 0, 3, 1.0, 0.0, BORDER_CONSTANT);
    opencv_time += timer.out("sobel_dx");

    Sobel(img16, sobel_dy, CV_16S, 0, 1, 3, 1.0, 0.0, BORDER_CONSTANT);
    opencv_time += timer.out("sobel_dy");

    std::cout << "opencv sobel total time: " << opencv_time << std::endl;

    std::vector<cv::Mat> in_v;
    in_v.push_back(img16);
    std::vector<cv::Mat> out_v;
    Mat fusion_sobel_dx(img.size(), CV_16S, cv::Scalar(0));
    Mat fusion_sobel_dy(img.size(), CV_16S, cv::Scalar(0));
    out_v.push_back(fusion_sobel_dx);
    out_v.push_back(fusion_sobel_dy);

    simple_fusion::ProcessManager manager(16, 128);
    manager.nodes_.clear();
    manager.nodes_.push_back(std::make_shared<simple_fusion::Sobel1x3SxxSyxNode_16S_16S>());
    manager.nodes_.push_back(std::make_shared<simple_fusion::Sobel3x1SxySyyNode_16S_16S>());
    manager.arrange(img.rows, img.cols);

    timer.reset();
    manager.process(in_v, out_v);
    timer.out("fusion sobel total time");

    Mat sobel_diff_x = cv::abs(sobel_dx - fusion_sobel_dx);
    Mat sobel_diff_y = cv::abs(sobel_dy - fusion_sobel_dy);

    imshow("diff_x", sobel_diff_x > 0);
    imshow("diff_y", sobel_diff_y > 0);
//    waitKey(0);

    manager.nodes_.clear();
    manager.nodes_.push_back(std::make_shared<simple_fusion::Sobel1x3SxxSyxNode_16S_16S>());
    manager.nodes_.push_back(std::make_shared<simple_fusion::Sobel3x1SxySyyNode_16S_16S>());
    manager.nodes_.push_back(std::make_shared<simple_fusion::MagSqure1x1Node_16S_32S>());
    manager.arrange(img.rows, img.cols);

    out_v.clear();
    Mat fusion_mag(img.size(), CV_32S, cv::Scalar(0));
    out_v.push_back(fusion_mag);

    timer.reset();
    manager.process(in_v, out_v);
    timer.out("fusion mag total time");

    Mat opencv_mag;

    timer.reset();
    sobel_dx.convertTo(sobel_dx, CV_32S);
    sobel_dy.convertTo(sobel_dy, CV_32S);
    opencv_mag = sobel_dx.mul(sobel_dx)  + sobel_dy.mul(sobel_dy);
    double mag_time = timer.elapsed();
    std::cout << "opencv mag total time: " << opencv_time + mag_time << std::endl;

    Mat diff_mag = cv::abs(opencv_mag - fusion_mag);
    imshow("diff_mag", diff_mag > 0);
    waitKey(0);

    std::cout << "test end" << std::endl;
}

void sobel_mag_phase_quant_test()
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
    imshow("img", img);
    std::cout << "test img size: " << img.rows * img.cols << std::endl
              << std::endl;

    Timer timer;
    double opencv_time = 0;

    Mat img16;
    img.convertTo(img16, CV_16S);

    Mat sobel_dx, sobel_dy, sobel_ag;
    Sobel(img16, sobel_dx, CV_16S, 1, 0, 3, 1.0, 0.0, BORDER_CONSTANT);
    opencv_time += timer.out("sobel_dx");

    Sobel(img16, sobel_dy, CV_16S, 0, 1, 3, 1.0, 0.0, BORDER_CONSTANT);
    opencv_time += timer.out("sobel_dy");

    sobel_dx.convertTo(sobel_dx, CV_32S);
    sobel_dy.convertTo(sobel_dy, CV_32S);
    Mat opencv_mag = sobel_dx.mul(sobel_dx)  + sobel_dy.mul(sobel_dy);
    opencv_time += timer.out("opencv_mag");

    Mat sobel_dx_f, sobel_dy_f;
    sobel_dx.convertTo(sobel_dx_f, CV_32F);
    sobel_dy.convertTo(sobel_dy_f, CV_32F);

    timer.reset();
    const int thresh = 60;
    Mat opencv_angle;
    cv::phase(sobel_dx_f, sobel_dy_f, opencv_angle, true);
    opencv_time += timer.out("opencv phase");

    Mat_<unsigned char> quantized_unfiltered;
    opencv_angle.convertTo(quantized_unfiltered, CV_8U, 16.0 / 360.0);

    // Mask 16 buckets into 8 quantized orientations
    for (int r = 1; r < opencv_angle.rows - 1; ++r)
    {
        uchar *quant_r = quantized_unfiltered.ptr<uchar>(r);
        int32_t *mag_r = opencv_mag.ptr<int32_t>(r);
        for (int c = 1; c < opencv_angle.cols - 1; ++c)
        {
            if(mag_r[c] > thresh * thresh){
                quant_r[c] &= 7;
            }else{
                quant_r[c] = 0;
            }
        }
    }
    opencv_time += timer.out("opencv quant");

    std::cout << "opencv quant total time: " << opencv_time << std::endl;

    std::vector<cv::Mat> in_v;
    in_v.push_back(img16);
    std::vector<cv::Mat> out_v;
    Mat fusion_quant(img.size(), CV_8U, cv::Scalar(0));
    out_v.push_back(fusion_quant);

    simple_fusion::ProcessManager manager(16, 128);
    manager.nodes_.clear();
    manager.nodes_.push_back(std::make_shared<simple_fusion::Sobel1x3SxxSyxNode_16S_16S>());
    manager.nodes_.push_back(std::make_shared<simple_fusion::Sobel3x1SxySyyNode_16S_16S>());
    manager.nodes_.push_back(std::make_shared<simple_fusion::MagPhaseQuant1x1Node_16S_8U>(thresh*thresh));
    manager.arrange(img.rows, img.cols);

    timer.reset();
    manager.process(in_v, out_v);
    timer.out("fusion quant total time");

    Mat quant_diff = fusion_quant != quantized_unfiltered;
    imshow("quant diff", quant_diff);
//    waitKey(0);

    manager.nodes_.clear();
    manager.nodes_.push_back(std::make_shared<simple_fusion::Sobel1x3SxxSyxNode_16S_16S>());
    manager.nodes_.push_back(std::make_shared<simple_fusion::Sobel3x1SxySyyNode_16S_16S>());
    manager.nodes_.push_back(std::make_shared<simple_fusion::MagPhaseQuantShift1x1Node_16S_8U>(thresh*thresh));
    manager.arrange(img.rows, img.cols);

    opencv_angle.convertTo(quantized_unfiltered, CV_8U, 16.0 / 360.0);
    for (int r = 1; r < opencv_angle.rows - 1; ++r)
    {
        uchar *quant_r = quantized_unfiltered.ptr<uchar>(r);
        int32_t *mag_r = opencv_mag.ptr<int32_t>(r);
        for (int c = 1; c < opencv_angle.cols - 1; ++c)
        {
            if(mag_r[c] > thresh * thresh){
                quant_r[c] &= 7;
                quant_r[c] = 1 << quant_r[c];
            }else{
                quant_r[c] = 0;
            }
        }
    }
    Mat quant_shift_diff = fusion_quant != quantized_unfiltered;
    imshow("quant shift diff", quant_shift_diff);
    waitKey(0);

    std::cout << "test end" << std::endl;
}
int main()
{
//    gauss_test();
//    sobel_mag_test();
    sobel_mag_phase_quant_test();
    return 0;
}
