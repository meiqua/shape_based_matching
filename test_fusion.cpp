#include "line2Dup.h"
#include <memory>
#include <iostream>
#include <assert.h>
#include <chrono>
#include <map>
using namespace std;
using namespace cv;

static std::string prefix = "/home/rfjiang/shape_based_matching/test/";

struct FilterNode
{
    std::vector<cv::Mat> buffers;

    int num_buf = 1;
    int buffer_rows = 0;
    int buffer_cols = 0;
    int padded_rows = 0;
    int padded_cols = 0;

    int anchor_row = 0; // anchor: where topleft is in full img
    int anchor_col = 0;

    int prepared_row = 0; // where have been calculated in full img
    int prepared_col = 0;
    int parent = -1;

    std::string op_name;
    int op_type = CV_16U;
    int op_r, op_c;

    int simd_step = mipp::N<int16_t>();
    //    bool use_simd = false;
    bool use_simd = true;

    template <class T>
    T *ptr(int r, int c, int buf_idx = 0)
    {
        r -= anchor_row; // from full img to buffer img
        c -= anchor_col;
        //        assert(r >= 0 && c >= 0);
        //        r = r % buffer_rows;  // row is changed because of rolling buffer
        return &buffers[buf_idx].at<T>(r, c);
    }

    std::function<int(int, int, int, int)> simple_update; // update start_r end_r start_c end_c
    std::function<int(int, int, int, int)> simd_update;

    void backward_rc(std::vector<FilterNode> &nodes, int rows, int cols, int cur_padded_rows, int cur_padded_cols) // calculate paddings
    {
        if (rows > buffer_rows)
        {
            buffer_rows = rows;
            padded_rows = cur_padded_rows;
        }
        if (cols > buffer_cols)
        {
            buffer_cols = cols;
            padded_cols = cur_padded_cols;
        }
        if (parent >= 0)
            nodes[parent].backward_rc(nodes, buffer_rows + op_r - 1, cols + op_c - 1,
                                      cur_padded_rows + op_r / 2, cur_padded_cols + op_c / 2);
    }
};

void fusion_test()
{
    line2Dup::Detector detector(128, {4, 8});
    std::vector<std::string> ids;
    ids.push_back("test");
    detector.readClasses(ids, prefix + "case1/%s_templ.yaml");

    // angle & scale are saved here, fetched by match id
    auto infos = shape_based_matching::shapeInfo_producer::load_infos(prefix + "case1/test_info.yaml");

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
    static const int KERNEL_SIZE = 5;
    Mat smoothed;
    GaussianBlur(img, smoothed, Size(KERNEL_SIZE, KERNEL_SIZE), 0, 0, BORDER_REPLICATE);
    timer.out("GaussianBlur");

    Mat sobel_dx, sobel_dy, sobel_ag;
    Sobel(smoothed, sobel_dx, CV_32F, 1, 0, 3, 1.0, 0.0, BORDER_REPLICATE);
    timer.out("sobel_dx");

    Sobel(smoothed, sobel_dy, CV_32F, 0, 1, 3, 1.0, 0.0, BORDER_REPLICATE);
    timer.out("sobel_dy");

    Mat magnitude = sobel_dx.mul(sobel_dx) + sobel_dy.mul(sobel_dy);
    timer.out("magnitude");

    phase(sobel_dx, sobel_dy, sobel_ag, true);
    timer.out("sobel_ag");

    const int tileRows = 32;
    const int tileCols = 256;
    const int num_threads = 8;

    const int imgRows = img.rows;
    const int imgCols = img.cols;
    const int thread_rows_step = imgRows / num_threads;

    // gaussian coff quantization
    const int gauss_size = 5;
    const int gauss_quant_bit = 4; // should be larger if gauss_size is larger
    cv::Mat double_gauss = cv::getGaussianKernel(gauss_size, 0, CV_64F);
    int32_t gauss_knl_uint32[gauss_size] = {0};
    for (int i = 0; i < gauss_size; i++)
    {
        gauss_knl_uint32[i] = int32_t(double_gauss.at<double>(i, 0) * (1 << gauss_quant_bit));
    }

//#pragma omp parallel for num_threads(num_threads)
        for(int thread_i = 0; thread_i < num_threads; thread_i++){
            const int tile_start_rows = thread_i * thread_rows_step;
            const int tile_end_rows = tile_start_rows + thread_rows_step;
        }

    imshow("img", img);
    waitKey(0);

    std::cout << "test end" << std::endl
              << std::endl;
}

int main()
{
    fusion_test();
    return 0;
}
