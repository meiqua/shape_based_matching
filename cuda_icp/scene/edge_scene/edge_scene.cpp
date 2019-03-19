#include "edge_scene.h"

using namespace cv;
using namespace std;

// https://github.com/songyuncen/EdgesSubPix/blob/master/EdgesSubPix.cpp
const double scale = 128.0;  // sum of half Canny filter is 128

static void getCannyKernel(OutputArray _d, double alpha)
{
    int r = cvRound(alpha * 3);
    int ksize = 2 * r + 1;

    _d.create(ksize, 1, CV_16S, -1, true);

    Mat k = _d.getMat();

    vector<float> kerF(ksize, 0.0f);
    kerF[r] = 0.0f;
    double a2 = alpha * alpha;
    float sum = 0.0f;
    for (int x = 1; x <= r; ++x)
    {
        float v = (float)(-x * std::exp(-x * x / (2 * a2)));
        sum += v;
        kerF[r + x] = v;
        kerF[r - x] = -v;
    }
    float scale = 128 / sum;
    for (int i = 0; i < ksize; ++i)
    {
        kerF[i] *= scale;
    }
    Mat temp(ksize, 1, CV_32F, &kerF[0]);
    temp.convertTo(k, CV_16S);
}

// non-maximum supression and hysteresis
static void postCannyFilter(const Mat &src, Mat &dx, Mat &dy, int low, int high, Mat &dst)
{
    ptrdiff_t mapstep = src.cols + 2;
    AutoBuffer<uchar> buffer((src.cols + 2)*(src.rows + 2) + mapstep * 3 * sizeof(int));

    // L2Gradient comparison with square
    high = high * high;
    low = low * low;

    int* mag_buf[3];
    mag_buf[0] = (int*)(uchar*)buffer;
    mag_buf[1] = mag_buf[0] + mapstep;
    mag_buf[2] = mag_buf[1] + mapstep;
    memset(mag_buf[0], 0, mapstep*sizeof(int));

    uchar* map = (uchar*)(mag_buf[2] + mapstep);
    memset(map, 1, mapstep);
    memset(map + mapstep*(src.rows + 1), 1, mapstep);

    int maxsize = std::max(1 << 10, src.cols * src.rows / 10);
    std::vector<uchar*> stack(maxsize);
    uchar **stack_top = &stack[0];
    uchar **stack_bottom = &stack[0];

    /* sector numbers
    (Top-Left Origin)
    1   2   3
    *  *  *
    * * *
    0*******0
    * * *
    *  *  *
    3   2   1
    */

#define CANNY_PUSH(d)    *(d) = uchar(2), *stack_top++ = (d)
#define CANNY_POP(d)     (d) = *--stack_top

#if CV_SSE2
    bool haveSSE2 = checkHardwareSupport(CV_CPU_SSE2);
#endif

    // calculate magnitude and angle of gradient, perform non-maxima suppression.
    // fill the map with one of the following values:
    //   0 - the pixel might belong to an edge
    //   1 - the pixel can not belong to an edge
    //   2 - the pixel does belong to an edge
    for (int i = 0; i <= src.rows; i++)
    {
        int* _norm = mag_buf[(i > 0) + 1] + 1;
        if (i < src.rows)
        {
            short* _dx = dx.ptr<short>(i);
            short* _dy = dy.ptr<short>(i);

            int j = 0, width = src.cols;
#if CV_SSE2
            if (haveSSE2)
            {
                for (; j <= width - 8; j += 8)
                {
                    __m128i v_dx = _mm_loadu_si128((const __m128i *)(_dx + j));
                    __m128i v_dy = _mm_loadu_si128((const __m128i *)(_dy + j));

                    __m128i v_dx_ml = _mm_mullo_epi16(v_dx, v_dx), v_dx_mh = _mm_mulhi_epi16(v_dx, v_dx);
                    __m128i v_dy_ml = _mm_mullo_epi16(v_dy, v_dy), v_dy_mh = _mm_mulhi_epi16(v_dy, v_dy);

                    __m128i v_norm = _mm_add_epi32(_mm_unpacklo_epi16(v_dx_ml, v_dx_mh), _mm_unpacklo_epi16(v_dy_ml, v_dy_mh));
                    _mm_storeu_si128((__m128i *)(_norm + j), v_norm);

                    v_norm = _mm_add_epi32(_mm_unpackhi_epi16(v_dx_ml, v_dx_mh), _mm_unpackhi_epi16(v_dy_ml, v_dy_mh));
                    _mm_storeu_si128((__m128i *)(_norm + j + 4), v_norm);
                }
            }
#elif CV_NEON
            for (; j <= width - 8; j += 8)
            {
                int16x8_t v_dx = vld1q_s16(_dx + j), v_dy = vld1q_s16(_dy + j);
                int16x4_t v_dxp = vget_low_s16(v_dx), v_dyp = vget_low_s16(v_dy);
                int32x4_t v_dst = vmlal_s16(vmull_s16(v_dxp, v_dxp), v_dyp, v_dyp);
                vst1q_s32(_norm + j, v_dst);

                v_dxp = vget_high_s16(v_dx), v_dyp = vget_high_s16(v_dy);
                v_dst = vmlal_s16(vmull_s16(v_dxp, v_dxp), v_dyp, v_dyp);
                vst1q_s32(_norm + j + 4, v_dst);
            }
#endif
            for (; j < width; ++j)
                _norm[j] = int(_dx[j])*_dx[j] + int(_dy[j])*_dy[j];

            _norm[-1] = _norm[src.cols] = 0;
        }
        else
            memset(_norm - 1, 0, /* cn* */mapstep*sizeof(int));

        // at the very beginning we do not have a complete ring
        // buffer of 3 magnitude rows for non-maxima suppression
        if (i == 0)
            continue;

        uchar* _map = map + mapstep*i + 1;
        _map[-1] = _map[src.cols] = 1;

        int* _mag = mag_buf[1] + 1; // take the central row
        ptrdiff_t magstep1 = mag_buf[2] - mag_buf[1];
        ptrdiff_t magstep2 = mag_buf[0] - mag_buf[1];

        const short* _x = dx.ptr<short>(i - 1);
        const short* _y = dy.ptr<short>(i - 1);

        if ((stack_top - stack_bottom) + src.cols > maxsize)
        {
            int sz = (int)(stack_top - stack_bottom);
            maxsize = std::max(maxsize * 3 / 2, sz + src.cols);
            stack.resize(maxsize);
            stack_bottom = &stack[0];
            stack_top = stack_bottom + sz;
        }

        int prev_flag = 0;
        for (int j = 0; j < src.cols; j++)
        {
            #define CANNY_SHIFT 15
            const int TG22 = (int)(0.4142135623730950488016887242097*(1 << CANNY_SHIFT) + 0.5);

            int m = _mag[j];

            if (m > low)
            {
                int xs = _x[j];
                int ys = _y[j];
                int x = std::abs(xs);
                int y = std::abs(ys) << CANNY_SHIFT;

                int tg22x = x * TG22;

                if (y < tg22x)
                {
                    if (m > _mag[j - 1] && m >= _mag[j + 1]) goto __ocv_canny_push;
                }
                else
                {
                    int tg67x = tg22x + (x << (CANNY_SHIFT + 1));
                    if (y > tg67x)
                    {
                        if (m > _mag[j + magstep2] && m >= _mag[j + magstep1]) goto __ocv_canny_push;
                    }
                    else
                    {
                        int s = (xs ^ ys) < 0 ? -1 : 1;
                        if (m > _mag[j + magstep2 - s] && m > _mag[j + magstep1 + s]) goto __ocv_canny_push;
                    }
                }
            }
            prev_flag = 0;
            _map[j] = uchar(1);
            continue;
        __ocv_canny_push:
            if (!prev_flag && m > high && _map[j - mapstep] != 2)
            {
                CANNY_PUSH(_map + j);
                prev_flag = 1;
            }
            else
                _map[j] = 0;
        }

        // scroll the ring buffer
        _mag = mag_buf[0];
        mag_buf[0] = mag_buf[1];
        mag_buf[1] = mag_buf[2];
        mag_buf[2] = _mag;
    }

    // now track the edges (hysteresis thresholding)
    while (stack_top > stack_bottom)
    {
        uchar* m;
        if ((stack_top - stack_bottom) + 8 > maxsize)
        {
            int sz = (int)(stack_top - stack_bottom);
            maxsize = maxsize * 3 / 2;
            stack.resize(maxsize);
            stack_bottom = &stack[0];
            stack_top = stack_bottom + sz;
        }

        CANNY_POP(m);

        if (!m[-1])         CANNY_PUSH(m - 1);
        if (!m[1])          CANNY_PUSH(m + 1);
        if (!m[-mapstep - 1]) CANNY_PUSH(m - mapstep - 1);
        if (!m[-mapstep])   CANNY_PUSH(m - mapstep);
        if (!m[-mapstep + 1]) CANNY_PUSH(m - mapstep + 1);
        if (!m[mapstep - 1])  CANNY_PUSH(m + mapstep - 1);
        if (!m[mapstep])    CANNY_PUSH(m + mapstep);
        if (!m[mapstep + 1])  CANNY_PUSH(m + mapstep + 1);
    }

    // the final pass, form the final image
    const uchar* pmap = map + mapstep + 1;
    uchar* pdst = dst.ptr();
    for (int i = 0; i < src.rows; i++, pmap += mapstep, pdst += dst.step)
    {
        for (int j = 0; j < src.cols; j++)
            pdst[j] = (uchar)-(pmap[j] >> 1);
    }
}

static inline  double getAmplitude(Mat &dx, Mat &dy, int i, int j)
{
    Point2d mag(dx.at<short>(i, j), dy.at<short>(i, j));
    return norm(mag);
}

static inline void getMagNeighbourhood(Mat &dx, Mat &dy, Point &p, int w, int h, vector<double> &mag)
{
    int top = p.y - 1 >= 0 ? p.y - 1 : p.y;
    int down = p.y + 1 < h ? p.y + 1 : p.y;
    int left = p.x - 1 >= 0 ? p.x - 1 : p.x;
    int right = p.x + 1 < w ? p.x + 1 : p.x;

    mag[0] = getAmplitude(dx, dy, top, left);
    mag[1] = getAmplitude(dx, dy, top, p.x);
    mag[2] = getAmplitude(dx, dy, top, right);
    mag[3] = getAmplitude(dx, dy, p.y, left);
    mag[4] = getAmplitude(dx, dy, p.y, p.x);
    mag[5] = getAmplitude(dx, dy, p.y, right);
    mag[6] = getAmplitude(dx, dy, down, left);
    mag[7] = getAmplitude(dx, dy, down, p.x);
    mag[8] = getAmplitude(dx, dy, down, right);
}

static inline void get2ndFacetModelIn3x3(vector<double> &mag, vector<double> &a)
{
    a[0] = (-mag[0] + 2.0 * mag[1] - mag[2] + 2.0 * mag[3] + 5.0 * mag[4] + 2.0 * mag[5] - mag[6] + 2.0 * mag[7] - mag[8]) / 9.0;
    a[1] = (-mag[0] + mag[2] - mag[3] + mag[5] - mag[6] + mag[8]) / 6.0;
    a[2] = (mag[6] + mag[7] + mag[8] - mag[0] - mag[1] - mag[2]) / 6.0;
    a[3] = (mag[0] - 2.0 * mag[1] + mag[2] + mag[3] - 2.0 * mag[4] + mag[5] + mag[6] - 2.0 * mag[7] + mag[8]) / 6.0;
    a[4] = (-mag[0] + mag[2] + mag[6] - mag[8]) / 4.0;
    a[5] = (mag[0] + mag[1] + mag[2] - 2.0 * (mag[3] + mag[4] + mag[5]) + mag[6] + mag[7] + mag[8]) / 6.0;
}
/*
   Compute the eigenvalues and eigenvectors of the Hessian matrix given by
   dfdrr, dfdrc, and dfdcc, and sort them in descending order according to
   their absolute values.
*/
static inline void eigenvals(vector<double> &a, double eigval[2], double eigvec[2][2])
{
    // derivatives
    // fx = a[1], fy = a[2]
    // fxy = a[4]
    // fxx = 2 * a[3]
    // fyy = 2 * a[5]
    double dfdrc = a[4];
    double dfdcc = a[3] * 2.0;
    double dfdrr = a[5] * 2.0;
    double theta, t, c, s, e1, e2, n1, n2; /* , phi; */

    /* Compute the eigenvalues and eigenvectors of the Hessian matrix. */
    if (dfdrc != 0.0) {
        theta = 0.5*(dfdcc - dfdrr) / dfdrc;
        t = 1.0 / (fabs(theta) + sqrt(theta*theta + 1.0));
        if (theta < 0.0) t = -t;
        c = 1.0 / sqrt(t*t + 1.0);
        s = t*c;
        e1 = dfdrr - t*dfdrc;
        e2 = dfdcc + t*dfdrc;
    }
    else {
        c = 1.0;
        s = 0.0;
        e1 = dfdrr;
        e2 = dfdcc;
    }
    n1 = c;
    n2 = -s;

    /* If the absolute value of an eigenvalue is larger than the other, put that
    eigenvalue into first position.  If both are of equal absolute value, put
    the negative one first. */
    if (fabs(e1) > fabs(e2)) {
        eigval[0] = e1;
        eigval[1] = e2;
        eigvec[0][0] = n1;
        eigvec[0][1] = n2;
        eigvec[1][0] = -n2;
        eigvec[1][1] = n1;
    }
    else if (fabs(e1) < fabs(e2)) {
        eigval[0] = e2;
        eigval[1] = e1;
        eigvec[0][0] = -n2;
        eigvec[0][1] = n1;
        eigvec[1][0] = n1;
        eigvec[1][1] = n2;
    }
    else {
        if (e1 < e2) {
            eigval[0] = e1;
            eigval[1] = e2;
            eigvec[0][0] = n1;
            eigvec[0][1] = n2;
            eigvec[1][0] = -n2;
            eigvec[1][1] = n1;
        }
        else {
            eigval[0] = e2;
            eigval[1] = e1;
            eigvec[0][0] = -n2;
            eigvec[0][1] = n1;
            eigvec[1][0] = n1;
            eigvec[1][1] = n2;
        }
    }
}

// end https://github.com/songyuncen/EdgesSubPix/blob/master/EdgesSubPix.cpp

template<class T>
T pow2(const T& in){return in*in;}

void Scene_edge::init_Scene_edge_cpu(cv::Mat img, std::vector<::Vec2f> &pcd_buffer,
                                     std::vector<::Vec2f>& normal_buffer, float max_dist_diff)
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

    double alpha = 1;
    int low = 30;
    int high = 60;

    Mat blur;
    GaussianBlur(gray, blur, Size(5, 5), alpha, alpha);

    Mat d;
    getCannyKernel(d, alpha);
    Mat one = Mat::ones(Size(1, 1), CV_16S);
    Mat dx, dy;
    sepFilter2D(blur, dx, CV_16S, d, one);
    sepFilter2D(blur, dy, CV_16S, one, d);

    // non-maximum supression & hysteresis threshold
    Mat edge = Mat::zeros(gray.size(), CV_8UC1);
    int lowThresh = cvRound(scale * low);
    int highThresh = cvRound(scale * high);
    postCannyFilter(gray, dx, dy, lowThresh, highThresh, edge);

//    cv::imshow("edge", edge);
//    cv::waitKey(0);

    normal_buffer.clear();
    normal_buffer.resize(img.rows * img.cols);

    pcd_buffer.clear();
    pcd_buffer.resize(img.rows * img.cols, ::Vec2f(-1, -1)); // -1 indicate no edge around


    cv::Mat sobel_dx, sobel_dy, magnitude, sobel_ag;
    cv::Sobel(blur, sobel_dx, CV_32F, 1, 0, 3, 1.0, 0.0, cv::BORDER_REPLICATE);
    cv::Sobel(blur, sobel_dy, CV_32F, 0, 1, 3, 1.0, 0.0, cv::BORDER_REPLICATE);
    phase(sobel_dx, sobel_dy, sobel_ag, false);


    for(int r=0; r<img.rows; r++){
        for(int c=0; c<img.cols; c++){
            if(edge.at<uchar>(r, c) > 0){  // get normals & pcds at edge only

                int w = dx.cols;
                int h = dx.rows;
                Point icontour = {c, r};

                vector<double> magNeighbour(9);
                getMagNeighbourhood(dx, dy, icontour, w, h, magNeighbour);
                vector<double> a(9);
                get2ndFacetModelIn3x3(magNeighbour, a);

                // Hessian eigen vector
                double eigvec[2][2], eigval[2];
                eigenvals(a, eigval, eigvec);
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

                float theta = sobel_ag.at<float>(r, c);
                float test_nx = cos(theta);
                float test_ny = -sin(theta);

                normal_buffer[c + r*img.cols] = {float(test_nx), float(test_ny)};
                pcd_buffer[c +r*img.cols] = {float(c), float(r)};
            }
        }
    }


    // get pcd, dilute to neibor
    {
        // may padding to divid and parallel
        cv::Mat dist_buffer(img.size(), CV_32FC1, FLT_MAX);
        int kernel_size = int(max_dist_diff+0.5f);
        for(int r=0+kernel_size; r<img.rows - kernel_size; r++){
            for(int c=0+kernel_size; c<img.cols - kernel_size; c++){

                if(edge.at<uchar>(r, c) > 0){

                    auto pcd = pcd_buffer[c +r*img.cols];

                    for(int i=-kernel_size; i<=kernel_size; i++){
                        for(int j=-kernel_size; j<=kernel_size; j++){
                            if(i==0 && j==0) continue; // already set pcd

                            float dist_sq = pow2(i-(pcd.y-r))*pow2(j-(pcd.x-c));

                            // don't go too far
                            if(dist_sq > max_dist_diff*max_dist_diff) continue;

                            int new_r = r + i;
                            int new_c = c + j;

                            // if closer
                            if(dist_sq < dist_buffer.at<float>(new_r, new_c)){
                                pcd_buffer[new_c + new_r*img.cols] = pcd;
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


