#include "line2Dup.h"
#include <iostream>

using namespace std;
using namespace cv;

#include <chrono>
class Timer
{
public:
    Timer() : beg_(clock_::now()) {}
    void reset() { beg_ = clock_::now(); }
    double elapsed() const {
        return std::chrono::duration_cast<second_>
            (clock_::now() - beg_).count(); }
    void out(std::string message = ""){
        double t = elapsed();
        std::cout << message << "  elasped time:" << t << "s" << std::endl;
        reset();
    }
private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1> > second_;
    std::chrono::time_point<clock_> beg_;
};

namespace line2Dup
{
/**
 * \brief Get the label [0,8) of the single bit set in quantized.
 */
static inline int getLabel(int quantized)
{
    switch (quantized)
    {
    case (1 << 0):
        return 0;
    case (1 << 1):
        return 1;
    case (1 << 2):
        return 2;
    case (1 << 3):
        return 3;
    case (1 << 4):
        return 4;
    case (1 << 5):
        return 5;
    case (1 << 6):
        return 6;
    case (1 << 7):
        return 7;
    case (1 << 8):
        return 8;
    case (1 << 9):
        return 9;
    case (1 << 10):
        return 10;
    case (1 << 11):
        return 11;
    case (1 << 12):
        return 12;
    case (1 << 13):
        return 13;
    case (1 << 14):
        return 14;
    case (1 << 15):
        return 15;
    default:
        CV_Error(Error::StsBadArg, "Invalid value of quantized parameter");
        return -1; //avoid warning
    }
}

void Feature::read(const FileNode &fn)
{
    FileNodeIterator fni = fn.begin();
    fni >> x >> y >> label;
}

void Feature::write(FileStorage &fs) const
{
    fs << "[:" << x << y << label << "]";
}

void Template::read(const FileNode &fn)
{
    width = fn["width"];
    height = fn["height"];
    tl_x = fn["tl_x"];
    tl_y = fn["tl_y"];
    pyramid_level = fn["pyramid_level"];

    FileNode features_fn = fn["features"];
    features.resize(features_fn.size());
    FileNodeIterator it = features_fn.begin(), it_end = features_fn.end();
    for (int i = 0; it != it_end; ++it, ++i)
    {
        features[i].read(*it);
    }
}

void Template::write(FileStorage &fs) const
{
    fs << "width" << width;
    fs << "height" << height;
    fs << "tl_x" << tl_x;
    fs << "tl_y" << tl_y;
    fs << "pyramid_level" << pyramid_level;

    fs << "features"
       << "[";
    for (int i = 0; i < (int)features.size(); ++i)
    {
        features[i].write(fs);
    }
    fs << "]"; // features
}

static Rect cropTemplates(std::vector<Template> &templates)
{
    int min_x = std::numeric_limits<int>::max();
    int min_y = std::numeric_limits<int>::max();
    int max_x = std::numeric_limits<int>::min();
    int max_y = std::numeric_limits<int>::min();

    // First pass: find min/max feature x,y over all pyramid levels and modalities
    for (int i = 0; i < (int)templates.size(); ++i)
    {
        Template &templ = templates[i];

        for (int j = 0; j < (int)templ.features.size(); ++j)
        {
            int x = templ.features[j].x << templ.pyramid_level;
            int y = templ.features[j].y << templ.pyramid_level;
            min_x = std::min(min_x, x);
            min_y = std::min(min_y, y);
            max_x = std::max(max_x, x);
            max_y = std::max(max_y, y);
        }
    }

    /// @todo Why require even min_x, min_y?
    if (min_x % 2 == 1)
        --min_x;
    if (min_y % 2 == 1)
        --min_y;

    // Second pass: set width/height and shift all feature positions
    for (int i = 0; i < (int)templates.size(); ++i)
    {
        Template &templ = templates[i];
        templ.width = (max_x - min_x) >> templ.pyramid_level;
        templ.height = (max_y - min_y) >> templ.pyramid_level;
        templ.tl_x = min_x >> templ.pyramid_level;
        templ.tl_y = min_y  >> templ.pyramid_level;

        for (int j = 0; j < (int)templ.features.size(); ++j)
        {
            templ.features[j].x -= templ.tl_x;
            templ.features[j].y -= templ.tl_y;
        }
    }

    return Rect(min_x, min_y, max_x - min_x, max_y - min_y);
}

bool ColorGradientPyramid::selectScatteredFeatures(const std::vector<Candidate> &candidates,
                                                   std::vector<Feature> &features,
                                                   size_t num_features, float distance)
{
    features.clear();
    float distance_sq = distance * distance;
    int i = 0;
    while (features.size() < num_features)
    {
        Candidate c = candidates[i];

        // Add if sufficient distance away from any previously chosen feature
        bool keep = true;
        for (int j = 0; (j < (int)features.size()) && keep; ++j)
        {
            Feature f = features[j];
            keep = (c.f.x - f.x) * (c.f.x - f.x) + (c.f.y - f.y) * (c.f.y - f.y) >= distance_sq;
        }
        if (keep)
            features.push_back(c.f);

        if (++i == (int)candidates.size())
        {
            // Start back at beginning, and relax required distance
            i = 0;
            distance -= 1.0f;
            distance_sq = distance * distance;
            // if (distance < 3)
            // {
            //     // we don't want two features too close
            //     break;
            // }
        }
    }
    if (features.size() == num_features)
    {
        return true;
    }
    else
    {
        std::cout << "this templ has no enough features" << std::endl;
        return false;
    }
}

/****************************************************************************************\
*                                                         Color gradient ColorGradient                                                                        *
\****************************************************************************************/

void hysteresisGradient(Mat &magnitude, Mat &quantized_angle,
                        Mat &angle, float threshold)
{
    // Quantize 360 degree range of orientations into 16 buckets
    // Note that [0, 11.25), [348.75, 360) both get mapped in the end to label 0,
    // for stability of horizontal and vertical features.
    Mat_<unsigned short> quantized_unfiltered;
    angle.convertTo(quantized_unfiltered, CV_16U, 32.0 / 360.0);

    // Zero out first and last columns
    for (int r = 0; r < quantized_unfiltered.cols; ++r)
    {
        quantized_unfiltered(0, r) = 0;
        quantized_unfiltered(quantized_unfiltered.rows - 1, r) = 0;
    }
    for (int r = 0; r < quantized_unfiltered.rows; ++r)
    {
        quantized_unfiltered(r, 0) = 0;
        quantized_unfiltered(r, quantized_unfiltered.cols - 1) = 0;
    }

    // Mask 32 buckets into 16 quantized orientations
    for (int r = 1; r < angle.rows-1; ++r)
    {
        ushort *quant_r = quantized_unfiltered.ptr<ushort>(r);
        for (int c = 1; c < angle.cols-1; ++c)
        {
            quant_r[c] &= 15;

        }
    }

    // Filter the raw quantized image. Only accept pixels where the magnitude is above some
    // threshold, and there is local agreement on the quantization.
    quantized_angle = Mat::zeros(angle.size(), CV_16U);
    for (int r = 1; r < angle.rows - 1; ++r)
    {
        float *mag_r = magnitude.ptr<float>(r);

        for (int c = 1; c < angle.cols - 1; ++c)
        {
            if (mag_r[c] > threshold)
            {
                // Compute histogram of quantized bins in 3x3 patch around pixel
                int histogram[16] = {0};

                ushort *patch3x3_row = &quantized_unfiltered(r - 1, c - 1);
                histogram[patch3x3_row[0]]++;
                histogram[patch3x3_row[1]]++;
                histogram[patch3x3_row[2]]++;

                patch3x3_row += quantized_unfiltered.step1();
                histogram[patch3x3_row[0]]++;
                histogram[patch3x3_row[1]]++;
                histogram[patch3x3_row[2]]++;

                patch3x3_row += quantized_unfiltered.step1();
                histogram[patch3x3_row[0]]++;
                histogram[patch3x3_row[1]]++;
                histogram[patch3x3_row[2]]++;

                // Find bin with the most votes from the patch
                int max_votes = 0;
                int index = -1;
                for (int i = 0; i < 16; ++i)
                {
                    if (max_votes < histogram[i])
                    {
                        index = i;
                        max_votes = histogram[i];
                    }
                }

                // Only accept the quantization if majority of pixels in the patch agree
                static const int NEIGHBOR_THRESHOLD = 5;
                if (max_votes >= NEIGHBOR_THRESHOLD)
                    quantized_angle.at<ushort>(r, c) = ushort(1 << index);
            }
        }
    }
}

static void quantizedOrientations(const Mat &src, Mat &magnitude,
                                  Mat &angle, float threshold)
{
    magnitude.create(src.size(), CV_32F);

    // Allocate temporary buffers
    Size size = src.size();
    Mat sobel_3dx;              // per-channel horizontal derivative
    Mat sobel_3dy;              // per-channel vertical derivative
    Mat sobel_dx(size, CV_32F); // maximum horizontal derivative
    Mat sobel_dy(size, CV_32F); // maximum vertical derivative
    Mat sobel_ag;               // final gradient orientation (unquantized)
    Mat smoothed;

    // Compute horizontal and vertical image derivatives on all color channels separately
    static const int KERNEL_SIZE = 7;
    // For some reason cvSmooth/cv::GaussianBlur, cvSobel/cv::Sobel have different defaults for border handling...
    GaussianBlur(src, smoothed, Size(KERNEL_SIZE, KERNEL_SIZE), 0, 0, BORDER_REPLICATE);
    Sobel(smoothed, sobel_3dx, CV_16S, 1, 0, 3, 1.0, 0.0, BORDER_REPLICATE);
    Sobel(smoothed, sobel_3dy, CV_16S, 0, 1, 3, 1.0, 0.0, BORDER_REPLICATE);

    short *ptrx = (short *)sobel_3dx.data;
    short *ptry = (short *)sobel_3dy.data;
    float *ptr0x = (float *)sobel_dx.data;
    float *ptr0y = (float *)sobel_dy.data;
    float *ptrmg = (float *)magnitude.data;

    const int length1 = static_cast<const int>(sobel_3dx.step1());
    const int length2 = static_cast<const int>(sobel_3dy.step1());
    const int length3 = static_cast<const int>(sobel_dx.step1());
    const int length4 = static_cast<const int>(sobel_dy.step1());
    const int length5 = static_cast<const int>(magnitude.step1());
    const int length0 = sobel_3dy.cols * 3;

    for (int r = 0; r < sobel_3dy.rows; ++r)
    {
        int ind = 0;

        for (int i = 0; i < length0; i += 3)
        {
            // Use the gradient orientation of the channel whose magnitude is largest
            int mag1 = ptrx[i + 0] * ptrx[i + 0] + ptry[i + 0] * ptry[i + 0];
            int mag2 = ptrx[i + 1] * ptrx[i + 1] + ptry[i + 1] * ptry[i + 1];
            int mag3 = ptrx[i + 2] * ptrx[i + 2] + ptry[i + 2] * ptry[i + 2];

            if (mag1 >= mag2 && mag1 >= mag3)
            {
                ptr0x[ind] = ptrx[i];
                ptr0y[ind] = ptry[i];
                ptrmg[ind] = (float)mag1;
            }
            else if (mag2 >= mag1 && mag2 >= mag3)
            {
                ptr0x[ind] = ptrx[i + 1];
                ptr0y[ind] = ptry[i + 1];
                ptrmg[ind] = (float)mag2;
            }
            else
            {
                ptr0x[ind] = ptrx[i + 2];
                ptr0y[ind] = ptry[i + 2];
                ptrmg[ind] = (float)mag3;
            }
            ++ind;
        }
        ptrx += length1;
        ptry += length2;
        ptr0x += length3;
        ptr0y += length4;
        ptrmg += length5;
    }

    // Calculate the final gradient orientations
    phase(sobel_dx, sobel_dy, sobel_ag, true);
    hysteresisGradient(magnitude, angle, sobel_ag, threshold * threshold);
}

ColorGradientPyramid::ColorGradientPyramid(const Mat &_src, const Mat &_mask,
                                           float _weak_threshold, size_t _num_features,
                                           float _strong_threshold)
    : src(_src),
      mask(_mask),
      pyramid_level(0),
      weak_threshold(_weak_threshold),
      num_features(_num_features),
      strong_threshold(_strong_threshold)
{
    update();
}

void ColorGradientPyramid::update()
{
    quantizedOrientations(src, magnitude, angle, weak_threshold);
}

void ColorGradientPyramid::pyrDown()
{
    // Some parameters need to be adjusted
    num_features /= 2; /// @todo Why not 4?
    ++pyramid_level;

    // Downsample the current inputs
    Size size(src.cols / 2, src.rows / 2);
    Mat next_src;
    cv::pyrDown(src, next_src, size);
    src = next_src;

    if (!mask.empty())
    {
        Mat next_mask;
        resize(mask, next_mask, size, 0.0, 0.0, INTER_NEAREST);
        mask = next_mask;
    }

    update();
}

void ColorGradientPyramid::quantize(Mat &dst) const
{
    dst = Mat::zeros(angle.size(), CV_16U);
    angle.copyTo(dst, mask);
}

bool ColorGradientPyramid::extractTemplate(Template &templ) const
{
    // Want features on the border to distinguish from background
    Mat local_mask;
    if (!mask.empty())
    {
        erode(mask, local_mask, Mat(), Point(-1, -1), 1, BORDER_REPLICATE);
//        subtract(mask, local_mask, local_mask);
    }

    std::vector<Candidate> candidates;
    bool no_mask = local_mask.empty();
    float threshold_sq = strong_threshold * strong_threshold;

    int nms_kernel_size = 5;
    cv::Mat magnitude_valid = cv::Mat(magnitude.size(), CV_8UC1, cv::Scalar(255));

    for (int r = 0+nms_kernel_size/2; r < magnitude.rows-nms_kernel_size/2; ++r)
    {
        const uchar *mask_r = no_mask ? NULL : local_mask.ptr<uchar>(r);

        for (int c = 0+nms_kernel_size/2; c < magnitude.cols-nms_kernel_size/2; ++c)
        {
            if (no_mask || mask_r[c])
            {
                float score = 0;
                if(magnitude_valid.at<uchar>(r, c)>0){
                    score = magnitude.at<float>(r, c);
                    bool is_max = true;
                    for(int r_offset = -nms_kernel_size/2; r_offset <= nms_kernel_size/2; r_offset++){
                        for(int c_offset = -nms_kernel_size/2; c_offset <= nms_kernel_size/2; c_offset++){
                            if(r_offset == 0 && c_offset == 0) continue;

                            if(score < magnitude.at<float>(r+r_offset, c+c_offset)){
                                score = 0;
                                is_max = false;
                                break;
                            }
                        }
                    }

                    if(is_max){
                        for(int r_offset = -nms_kernel_size/2; r_offset <= nms_kernel_size/2; r_offset++){
                            for(int c_offset = -nms_kernel_size/2; c_offset <= nms_kernel_size/2; c_offset++){
                                if(r_offset == 0 && c_offset == 0) continue;
                                magnitude_valid.at<uchar>(r+r_offset, c+c_offset) = 0;
                            }
                        }
                    }
                }

                if (score > threshold_sq && angle.at<ushort>(r, c)>0)
                {
                    candidates.push_back(Candidate(c, r, getLabel(angle.at<ushort>(r, c)), score));
                }
            }
        }
    }
    // We require a certain number of features
    if (candidates.size() < num_features)
        return false;
    // NOTE: Stable sort to agree with old code, which used std::list::sort()
    std::stable_sort(candidates.begin(), candidates.end());

    // Use heuristic based on surplus of candidates in narrow outline for initial distance threshold
    float distance = static_cast<float>(candidates.size() / num_features + 1);
    if (!selectScatteredFeatures(candidates, templ.features, num_features, distance))
    {
        return false;
    }

    // Size determined externally, needs to match templates for other modalities
    templ.width = -1;
    templ.height = -1;
    templ.pyramid_level = pyramid_level;

    return true;
}

ColorGradient::ColorGradient()
    : weak_threshold(10.0f),
      num_features(63),
      strong_threshold(55.0f)
{
}

ColorGradient::ColorGradient(float _weak_threshold, size_t _num_features, float _strong_threshold)
    : weak_threshold(_weak_threshold),
      num_features(_num_features),
      strong_threshold(_strong_threshold)
{
}

static const char CG_NAME[] = "ColorGradient";

std::string ColorGradient::name() const
{
    return CG_NAME;
}

void ColorGradient::read(const FileNode &fn)
{
    String type = fn["type"];
    CV_Assert(type == CG_NAME);

    weak_threshold = fn["weak_threshold"];
    num_features = int(fn["num_features"]);
    strong_threshold = fn["strong_threshold"];
}

void ColorGradient::write(FileStorage &fs) const
{
    fs << "type" << CG_NAME;
    fs << "weak_threshold" << weak_threshold;
    fs << "num_features" << int(num_features);
    fs << "strong_threshold" << strong_threshold;
}
/****************************************************************************************\
*                                                                 Response maps                                                                                    *
\****************************************************************************************/

static void orUnaligned16u(const ushort *src, const int src_stride,
                          ushort *dst, const int dst_stride,
                          const int width, const int height)
{
    for (int r = 0; r < height; ++r)
    {
        int c = 0;

        for (; c < width; ++c)
            dst[c] |= src[c];

        src += src_stride;
        dst += dst_stride;
    }
}

static void spread(const Mat &src, Mat &dst, int T)
{
    // Allocate and zero-initialize spread (OR'ed) image
    dst = Mat::zeros(src.size(), CV_16U);

    // Fill in spread gradient image (section 2.3)
    for (int r = 0; r < T; ++r)
    {
        int height = src.rows - r;
        for (int c = 0; c < T; ++c)
        {
            orUnaligned16u(&src.at<ushort>(r, c), static_cast<const int>(src.step1()), dst.ptr<ushort>(),
                          static_cast<const int>(dst.step1()), src.cols - c, height);
        }
    }
}

CV_DECL_ALIGNED(16) static const unsigned char SIMILARITY_LUT[1024] = {
0, 4, 4, 4, 4, 4, 4, 4, 1, 4, 4, 4, 4, 4, 4, 4, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4,
0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 4, 1, 4, 1, 4, 1, 4, 0, 4, 1, 4, 1, 4, 1, 4,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
0, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 1, 4, 4, 4, 1, 4, 4, 4, 1, 4, 4, 4,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
0, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 4, 1, 4, 4, 4, 4, 4, 4, 4,
0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
0, 4, 1, 4, 1, 4, 1, 4, 0, 4, 1, 4, 1, 4, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
0, 4, 4, 4, 1, 4, 4, 4, 1, 4, 4, 4, 1, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
0, 4, 4, 4, 4, 4, 4, 4, 1, 4, 4, 4, 4, 4, 4, 4, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4,
0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 4, 1, 4, 1, 4, 1, 4, 0, 4, 1, 4, 1, 4, 1, 4,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
0, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 1, 4, 4, 4, 1, 4, 4, 4, 1, 4, 4, 4,
0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 4, 1, 4, 4, 4, 4, 4, 4, 4,
0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
0, 4, 1, 4, 1, 4, 1, 4, 0, 4, 1, 4, 1, 4, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
0, 4, 4, 4, 1, 4, 4, 4, 1, 4, 4, 4, 1, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4};



static void computeResponseMaps(const Mat &src, std::vector<Mat> &response_maps)
{
    CV_Assert((src.rows * src.cols) % 16 == 0);

    // Allocate response maps
    response_maps.resize(16);
    for (int i = 0; i < 16; ++i)
        response_maps[i].create(src.size(), CV_8U);

    // 4*4bits
    Mat ___o(src.size(), CV_8U);
    Mat __o_(src.size(), CV_8U);
    Mat _o__(src.size(), CV_8U);
    Mat o___(src.size(), CV_8U);

    for (int r = 0; r < src.rows; ++r)
    {
        const ushort *src_r = src.ptr<ushort>(r);
        uchar *___o_r = ___o.ptr(r);
        uchar *__o__r = __o_.ptr(r);
        uchar *_o___r = _o__.ptr(r);
        uchar *o____r = o___.ptr(r);

        for (int c = 0; c < src.cols; ++c)
        {
            ___o_r[c] = src_r[c] & 15;
            __o__r[c] = (src_r[c] & (15 << 4)) >> 4;
            _o___r[c] = (src_r[c] & (15 << 8)) >> 8;
            o____r[c] = (src_r[c] & (15 << 16)) >> 16;
        }
    }

#if CV_SSSE3
    volatile bool haveSSSE3 = checkHardwareSupport(CV_CPU_SSSE3);
    if (haveSSSE3)
    {
        const __m128i *lut = reinterpret_cast<const __m128i *>(SIMILARITY_LUT);
        for (int ori = 0; ori < 16; ++ori)
        {
            __m128i *map_data = response_maps[ori].ptr<__m128i>();

            __m128i *___o_data = ___o.ptr<__m128i>();
            __m128i *__o__data = __o_.ptr<__m128i>();
            __m128i *_o___data = _o__.ptr<__m128i>();
            __m128i *o____data = o___.ptr<__m128i>();

            // Precompute the 2D response map S_i (section 2.4)
            for (int i = 0; i < (src.rows * src.cols) / 16; ++i)
            {
                // Using SSE shuffle for table lookup on 4 orientations at a time
                // The most/least significant 4 bits are used as the LUT index
                __m128i res1 = _mm_shuffle_epi8(lut[4 * ori + 0], ___o_data[i]);
                __m128i res2 = _mm_shuffle_epi8(lut[4 * ori + 1], __o__data[i]);
                __m128i res3 = _mm_shuffle_epi8(lut[4 * ori + 2], _o___data[i]);
                __m128i res4 = _mm_shuffle_epi8(lut[4 * ori + 3], o____data[i]);

                // Combine the results into a single similarity score
                __m128i res1_2 = _mm_max_epu8(res1, res2);
                __m128i res3_4 = _mm_max_epu8(res3, res4);
                map_data[i] = _mm_max_epu8(res1_2, res3_4);
            }
        }
    }
    else
#endif
    {
        // For each of the 8 quantized orientations...
        for (int ori = 0; ori < 16; ++ori)
        {
            uchar *map_data = response_maps[ori].ptr<uchar>();

            uchar *___o_data = ___o.ptr<uchar>();
            uchar *__o__data = __o_.ptr<uchar>();
            uchar *_o___data = _o__.ptr<uchar>();
            uchar *o____data = o___.ptr<uchar>();

            const uchar *lut_0 = SIMILARITY_LUT + 16* 4 * ori;
            const uchar *lut_1 = lut_0 + 16;
            const uchar *lut_2 = lut_1 + 16;
            const uchar *lut_3 = lut_2 + 16;

            for (int i = 0; i < src.rows * src.cols; ++i)
            {
                uchar max_01 = std::max(lut_0[___o_data[i]], lut_1[__o__data[i]]);
                uchar max_23 = std::max(lut_2[_o___data[i]], lut_3[o____data[i]]);
                map_data[i] = std::max(max_01, max_23);
            }
        }
    }
}

static void linearize(const Mat &response_map, Mat &linearized, int T)
{
    CV_Assert(response_map.rows % T == 0);
    CV_Assert(response_map.cols % T == 0);

    // linearized has T^2 rows, where each row is a linear memory
    int mem_width = response_map.cols / T;
    int mem_height = response_map.rows / T;
    linearized.create(T * T, mem_width * mem_height, CV_8U);

    // Outer two for loops iterate over top-left T^2 starting pixels
    int index = 0;
    for (int r_start = 0; r_start < T; ++r_start)
    {
        for (int c_start = 0; c_start < T; ++c_start)
        {
            uchar *memory = linearized.ptr(index);
            ++index;

            // Inner two loops copy every T-th pixel into the linear memory
            for (int r = r_start; r < response_map.rows; r += T)
            {
                const uchar *response_data = response_map.ptr(r);
                for (int c = c_start; c < response_map.cols; c += T)
                    *memory++ = response_data[c];
            }
        }
    }
}
/****************************************************************************************\
*                                                             Linearized similarities                                                                    *
\****************************************************************************************/

static const unsigned char *accessLinearMemory(const std::vector<Mat> &linear_memories,
                                               const Feature &f, int T, int W)
{
    // Retrieve the TxT grid of linear memories associated with the feature label
    const Mat &memory_grid = linear_memories[f.label];
    CV_DbgAssert(memory_grid.rows == T * T);
    CV_DbgAssert(f.x >= 0);
    CV_DbgAssert(f.y >= 0);
    // The LM we want is at (x%T, y%T) in the TxT grid (stored as the rows of memory_grid)
    int grid_x = f.x % T;
    int grid_y = f.y % T;
    int grid_index = grid_y * T + grid_x;
    CV_DbgAssert(grid_index >= 0);
    CV_DbgAssert(grid_index < memory_grid.rows);
    const unsigned char *memory = memory_grid.ptr(grid_index);
    // Within the LM, the feature is at (x/T, y/T). W is the "width" of the LM, the
    // input image width decimated by T.
    int lm_x = f.x / T;
    int lm_y = f.y / T;
    int lm_index = lm_y * W + lm_x;
    CV_DbgAssert(lm_index >= 0);
    CV_DbgAssert(lm_index < memory_grid.cols);
    return memory + lm_index;
}

static void similarity(const std::vector<Mat> &linear_memories, const Template &templ,
                       Mat &dst, Size size, int T)
{
    // we only have one modality, so 8192*2
    CV_Assert(templ.features.size() < 16384);

    // Decimate input image size by factor of T
    int W = size.width / T;
    int H = size.height / T;

    // Feature dimensions, decimated by factor T and rounded up
    int wf = (templ.width - 1) / T + 1;
    int hf = (templ.height - 1) / T + 1;

    // Span is the range over which we can shift the template around the input image
    int span_x = W - wf;
    int span_y = H - hf;

    int template_positions = span_y * W + span_x + 1; // why add 1?

    dst = Mat::zeros(H, W, CV_16U);
    short *dst_ptr = dst.ptr<short>();

#if CV_SSE2
    volatile bool haveSSE2 = checkHardwareSupport(CV_CPU_SSE2);
#endif

    for (int i = 0; i < (int)templ.features.size(); ++i)
    {

        Feature f = templ.features[i];

        if (f.x < 0 || f.x >= size.width || f.y < 0 || f.y >= size.height)
            continue;
        const uchar *lm_ptr = accessLinearMemory(linear_memories, f, T, W);

        // Now we do an aligned/unaligned add of dst_ptr and lm_ptr with template_positions elements
        int j = 0;
#if CV_SSE2
        if (haveSSE2)
        {
            __m128i const zero = _mm_setzero_si128();
            // Fall back to MOVDQU
            for (; j < template_positions - 7; j += 8)
            {
                __m128i responses = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(lm_ptr + j));
                __m128i *dst_ptr_sse = reinterpret_cast<__m128i *>(dst_ptr + j);
                responses = _mm_unpacklo_epi8(responses, zero);
                *dst_ptr_sse = _mm_add_epi16(*dst_ptr_sse, responses);
            }
        }
#endif
        for (; j < template_positions; ++j)
            dst_ptr[j] = short(dst_ptr[j] + short(lm_ptr[j]));
    }
}

static void similarityLocal(const std::vector<Mat> &linear_memories, const Template &templ,
                            Mat &dst, Size size, int T, Point center)
{
    CV_Assert(templ.features.size() < 16384);

    int W = size.width / T;
    dst = Mat::zeros(16, 16, CV_16U);

    int offset_x = (center.x / T - 8) * T;
    int offset_y = (center.y / T - 8) * T;

#if CV_SSE2
    volatile bool haveSSE2 = checkHardwareSupport(CV_CPU_SSE2);
    __m128i *dst_ptr_sse = dst.ptr<__m128i>();
#endif

    for (int i = 0; i < (int)templ.features.size(); ++i)
    {
        Feature f = templ.features[i];
        f.x += offset_x;
        f.y += offset_y;
        // Discard feature if out of bounds, possibly due to applying the offset
        if (f.x < 0 || f.y < 0 || f.x >= size.width || f.y >= size.height)
            continue;

        const uchar *lm_ptr = accessLinearMemory(linear_memories, f, T, W);
#if CV_SSE2
        if (haveSSE2)
        {
            __m128i const zero = _mm_setzero_si128();
            for (int row = 0; row < 16; ++row)
            {
                __m128i aligned_low = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(lm_ptr));
                __m128i aligned_high = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(lm_ptr + 8));
                aligned_low = _mm_unpacklo_epi8(aligned_low, zero);
                aligned_high = _mm_unpacklo_epi8(aligned_high, zero);
                dst_ptr_sse[2 * row] = _mm_add_epi16(dst_ptr_sse[2 * row], aligned_low);
                dst_ptr_sse[2 * row + 1] = _mm_add_epi16(dst_ptr_sse[2 * row + 1], aligned_high);
                lm_ptr += W; // Step to next row
            }
        }
        else
#endif
        {
            short *dst_ptr = dst.ptr<short>();
            for (int row = 0; row < 16; ++row)
            {
                for (int col = 0; col < 16; ++col)
                    dst_ptr[col] = short(dst_ptr[col] + short(lm_ptr[col]));
                dst_ptr += 16;
                lm_ptr += W;
            }
        }
    }
}

static void similarity_64(const std::vector<Mat> &linear_memories, const Template &templ,
                          Mat &dst, Size size, int T)
{
    // 63 features or less is a special case because the max similarity per-feature is 4.
    // 255/4 = 63, so up to that many we can add up similarities in 8 bits without worrying
    // about overflow. Therefore here we use _mm_add_epi8 as the workhorse, whereas a more
    // general function would use _mm_add_epi16.
    CV_Assert(templ.features.size() <= 63);
    /// @todo Handle more than 255/MAX_RESPONSE features!!

    // Decimate input image size by factor of T
    int W = size.width / T;
    int H = size.height / T;

    // Feature dimensions, decimated by factor T and rounded up
    int wf = (templ.width - 1) / T + 1;
    int hf = (templ.height - 1) / T + 1;

    // Span is the range over which we can shift the template around the input image
    int span_x = W - wf;
    int span_y = H - hf;

    // Compute number of contiguous (in memory) pixels to check when sliding feature over
    // image. This allows template to wrap around left/right border incorrectly, so any
    // wrapped template matches must be filtered out!
    int template_positions = span_y * W + span_x + 1; // why add 1?
    //int template_positions = (span_y - 1) * W + span_x; // More correct?

    /// @todo In old code, dst is buffer of size m_U. Could make it something like
    /// (span_x)x(span_y) instead?
    dst = Mat::zeros(H, W, CV_8U);
    uchar *dst_ptr = dst.ptr<uchar>();

#if CV_SSE2
    volatile bool haveSSE2 = checkHardwareSupport(CV_CPU_SSE2);
#if CV_SSE3
    volatile bool haveSSE3 = checkHardwareSupport(CV_CPU_SSE3);
#endif
#endif

    // Compute the similarity measure for this template by accumulating the contribution of
    // each feature
    for (int i = 0; i < (int)templ.features.size(); ++i)
    {
        // Add the linear memory at the appropriate offset computed from the location of
        // the feature in the template
        Feature f = templ.features[i];
        // Discard feature if out of bounds
        /// @todo Shouldn't actually see x or y < 0 here?
        if (f.x < 0 || f.x >= size.width || f.y < 0 || f.y >= size.height)
            continue;
        const uchar *lm_ptr = accessLinearMemory(linear_memories, f, T, W);

        // Now we do an aligned/unaligned add of dst_ptr and lm_ptr with template_positions elements
        int j = 0;
#if CV_SSE2
#if CV_SSE3
        if (haveSSE3)
        {
            // LDDQU may be more efficient than MOVDQU for unaligned load of next 16 responses
            for (; j < template_positions - 15; j += 16)
            {
                __m128i responses = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(lm_ptr + j));
                __m128i *dst_ptr_sse = reinterpret_cast<__m128i *>(dst_ptr + j);
                *dst_ptr_sse = _mm_add_epi8(*dst_ptr_sse, responses);
            }
        }
        else
#endif
            if (haveSSE2)
        {
            // Fall back to MOVDQU
            for (; j < template_positions - 15; j += 16)
            {
                __m128i responses = _mm_loadu_si128(reinterpret_cast<const __m128i *>(lm_ptr + j));
                __m128i *dst_ptr_sse = reinterpret_cast<__m128i *>(dst_ptr + j);
                *dst_ptr_sse = _mm_add_epi8(*dst_ptr_sse, responses);
            }
        }
#endif
        for (; j < template_positions; ++j)
            dst_ptr[j] = uchar(dst_ptr[j] + lm_ptr[j]);
    }
}

static void similarityLocal_64(const std::vector<Mat> &linear_memories, const Template &templ,
                               Mat &dst, Size size, int T, Point center)
{
    // Similar to whole-image similarity() above. This version takes a position 'center'
    // and computes the energy in the 16x16 patch centered on it.
    CV_Assert(templ.features.size() <= 63);

    // Compute the similarity map in a 16x16 patch around center
    int W = size.width / T;
    dst = Mat::zeros(16, 16, CV_8U);

    // Offset each feature point by the requested center. Further adjust to (-8,-8) from the
    // center to get the top-left corner of the 16x16 patch.
    // NOTE: We make the offsets multiples of T to agree with results of the original code.
    int offset_x = (center.x / T - 8) * T;
    int offset_y = (center.y / T - 8) * T;

#if CV_SSE2
    volatile bool haveSSE2 = checkHardwareSupport(CV_CPU_SSE2);
#if CV_SSE3
    volatile bool haveSSE3 = checkHardwareSupport(CV_CPU_SSE3);
#endif
    __m128i *dst_ptr_sse = dst.ptr<__m128i>();
#endif

    for (int i = 0; i < (int)templ.features.size(); ++i)
    {
        Feature f = templ.features[i];
        f.x += offset_x;
        f.y += offset_y;
        // Discard feature if out of bounds, possibly due to applying the offset
        if (f.x < 0 || f.y < 0 || f.x >= size.width || f.y >= size.height)
            continue;

        const uchar *lm_ptr = accessLinearMemory(linear_memories, f, T, W);
#if CV_SSE2
#if CV_SSE3
        if (haveSSE3)
        {
            // LDDQU may be more efficient than MOVDQU for unaligned load of 16 responses from current row
            for (int row = 0; row < 16; ++row)
            {
                __m128i aligned = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(lm_ptr));
                dst_ptr_sse[row] = _mm_add_epi8(dst_ptr_sse[row], aligned);
                lm_ptr += W; // Step to next row
            }
        }
        else
#endif
            if (haveSSE2)
        {
            // Fall back to MOVDQU
            for (int row = 0; row < 16; ++row)
            {
                __m128i aligned = _mm_loadu_si128(reinterpret_cast<const __m128i *>(lm_ptr));
                dst_ptr_sse[row] = _mm_add_epi8(dst_ptr_sse[row], aligned);
                lm_ptr += W; // Step to next row
            }
        }
        else
#endif
        {
            uchar *dst_ptr = dst.ptr<uchar>();
            for (int row = 0; row < 16; ++row)
            {
                for (int col = 0; col < 16; ++col)
                    dst_ptr[col] = uchar(dst_ptr[col] + lm_ptr[col]);
                dst_ptr += 16;
                lm_ptr += W;
            }
        }
    }
}

/****************************************************************************************\
*                                                             High-level Detector API                                                                    *
\****************************************************************************************/

Detector::Detector()
{
    this->modality = makePtr<ColorGradient>();
    pyramid_levels = 2;
    T_at_level.push_back(5);
    T_at_level.push_back(8);
}

Detector::Detector(std::vector<int> T)
{
    this->modality = makePtr<ColorGradient>();
    pyramid_levels = T.size();
    T_at_level = T;
}

Detector::Detector(int num_features, std::vector<int> T)
{
    this->modality = makePtr<ColorGradient>(10.0f, num_features, 55.0f);
    pyramid_levels = T.size();
    T_at_level = T;
}

std::vector<Match> Detector::match(Mat source, float threshold,
                                   const std::vector<std::string> &class_ids, const Mat mask) const
{
    Timer timer;
    std::vector<Match> matches;

    // Initialize each ColorGradient with our sources
    std::vector<Ptr<ColorGradientPyramid>> quantizers;
    CV_Assert(mask.empty() || mask.size() == source.size());
    quantizers.push_back(modality->process(source, mask));

    // pyramid level -> ColorGradient -> quantization
    LinearMemoryPyramid lm_pyramid(pyramid_levels,
                                   std::vector<LinearMemories>(1, LinearMemories(16)));

    // For each pyramid level, precompute linear memories for each ColorGradient
    std::vector<Size> sizes;
    for (int l = 0; l < pyramid_levels; ++l)
    {
        int T = T_at_level[l];
        std::vector<LinearMemories> &lm_level = lm_pyramid[l];

        if (l > 0)
        {
            for (int i = 0; i < (int)quantizers.size(); ++i)
                quantizers[i]->pyrDown();
        }

        Mat quantized, spread_quantized;
        std::vector<Mat> response_maps;
        for (int i = 0; i < (int)quantizers.size(); ++i)
        {
            quantizers[i]->quantize(quantized);
            spread(quantized, spread_quantized, T);
            computeResponseMaps(spread_quantized, response_maps);

            LinearMemories &memories = lm_level[i];
            for (int j = 0; j < 16; ++j)
                linearize(response_maps[j], memories[j], T);
        }

        sizes.push_back(quantized.size());
    }

    timer.out("construct response map");

    if (class_ids.empty())
    {
        // Match all templates
        TemplatesMap::const_iterator it = class_templates.begin(), itend = class_templates.end();
        for (; it != itend; ++it)
            matchClass(lm_pyramid, sizes, threshold, matches, it->first, it->second);
    }
    else
    {
        // Match only templates for the requested class IDs
        for (int i = 0; i < (int)class_ids.size(); ++i)
        {
            TemplatesMap::const_iterator it = class_templates.find(class_ids[i]);
            if (it != class_templates.end())
                matchClass(lm_pyramid, sizes, threshold, matches, it->first, it->second);
        }
    }

    // Sort matches by similarity, and prune any duplicates introduced by pyramid refinement
    std::sort(matches.begin(), matches.end());
    std::vector<Match>::iterator new_end = std::unique(matches.begin(), matches.end());
    matches.erase(new_end, matches.end());

    timer.out("templ match");

    return matches;
}

// Used to filter out weak matches
struct MatchPredicate
{
    MatchPredicate(float _threshold) : threshold(_threshold) {}
    bool operator()(const Match &m) { return m.similarity < threshold; }
    float threshold;
};

void Detector::matchClass(const LinearMemoryPyramid &lm_pyramid,
                          const std::vector<Size> &sizes,
                          float threshold, std::vector<Match> &matches,
                          const std::string &class_id,
                          const std::vector<TemplatePyramid> &template_pyramids) const
{
    for (size_t template_id = 0; template_id < template_pyramids.size(); ++template_id)
    {
        const TemplatePyramid &tp = template_pyramids[template_id];
        // First match over the whole image at the lowest pyramid level
        /// @todo Factor this out into separate function
        const std::vector<LinearMemories> &lowest_lm = lm_pyramid.back();

        // Compute similarity maps for each ColorGradient at lowest pyramid level
        Mat similarities;
        int lowest_start = static_cast<int>(tp.size() - 1);
        int lowest_T = T_at_level.back();
        int num_features = 0;
        int feature_64 = -1;
        {
            const Template &templ = tp[lowest_start];
            num_features += static_cast<int>(templ.features.size());
            if (feature_64 <= 0)
            {
                if (templ.features.size() < 64)
                {
                    feature_64 = 1;
                }
                else if (templ.features.size() < 16384)
                {
                    feature_64 = 2;
                }
            }
            if (feature_64 == 1)
            {
                similarity_64(lowest_lm[0], templ, similarities, sizes.back(), lowest_T);
            }
            else if (feature_64 == 2)
            {
                similarity(lowest_lm[0], templ, similarities, sizes.back(), lowest_T);
            }
        }

        if (feature_64 == 1)
        {
            similarities.convertTo(similarities, CV_16U);
        }

        // Find initial matches
        std::vector<Match> candidates;
        for (int r = 0; r < similarities.rows; ++r)
        {
            ushort *row = similarities.ptr<ushort>(r);
            for (int c = 0; c < similarities.cols; ++c)
            {
                int raw_score = row[c];
                float score = (raw_score * 100.f) / (4 * num_features);

                if (score > threshold)
                {
                    int offset = lowest_T / 2 + (lowest_T % 2 - 1);
                    int x = c * lowest_T + offset;
                    int y = r * lowest_T + offset;
                    candidates.push_back(Match(x, y, score, class_id, static_cast<int>(template_id)));
                }
            }
        }

        // Locally refine each match by marching up the pyramid
        for (int l = pyramid_levels - 2; l >= 0; --l)
        {
            const std::vector<LinearMemories> &lms = lm_pyramid[l];
            int T = T_at_level[l];
            int start = static_cast<int>(l);
            Size size = sizes[l];
            int border = 8 * T;
            int offset = T / 2 + (T % 2 - 1);
            int max_x = size.width - tp[start].width - border;
            int max_y = size.height - tp[start].height - border;

            Mat similarities2;
            for (int m = 0; m < (int)candidates.size(); ++m)
            {
                Match &match2 = candidates[m];
                int x = match2.x * 2 + 1; /// @todo Support other pyramid distance
                int y = match2.y * 2 + 1;

                // Require 8 (reduced) row/cols to the up/left
                x = std::max(x, border);
                y = std::max(y, border);

                // Require 8 (reduced) row/cols to the down/left, plus the template size
                x = std::min(x, max_x);
                y = std::min(y, max_y);

                // Compute local similarity maps for each ColorGradient
                int numFeatures = 0;
                feature_64 = -1;
                {
                    const Template &templ = tp[start];
                    numFeatures += static_cast<int>(templ.features.size());
                    if (feature_64 <= 0)
                    {
                        if (templ.features.size() < 64)
                        {
                            feature_64 = 1;
                        }
                        else if (templ.features.size() < 16384)
                        {
                            feature_64 = 2;
                        }
                    }
                    if (feature_64 == 1)
                    {
                        similarityLocal_64(lms[0], templ, similarities2, size, T, Point(x, y));
                    }
                    else if (feature_64 == 2)
                    {
                        similarityLocal(lms[0], templ, similarities2, size, T, Point(x, y));
                    }
                }

                if (feature_64 == 1)
                {
                    similarities2.convertTo(similarities2, CV_16U);
                }

                // Find best local adjustment
                float best_score = 0;
                int best_r = -1, best_c = -1;
                for (int r = 0; r < similarities2.rows; ++r)
                {
                    ushort *row = similarities2.ptr<ushort>(r);
                    for (int c = 0; c < similarities2.cols; ++c)
                    {
                        int score_int = row[c];
                        float score = (score_int * 100.f) / (4 * numFeatures);

                        if (score > best_score)
                        {
                            best_score = score;
                            best_r = r;
                            best_c = c;
                        }
                    }
                }
                // Update current match
                match2.similarity = best_score;
                match2.x = (x / T - 8 + best_c) * T + offset;
                match2.y = (y / T - 8 + best_r) * T + offset;
            }

            // Filter out any matches that drop below the similarity threshold
            std::vector<Match>::iterator new_end = std::remove_if(candidates.begin(), candidates.end(),
                                                                  MatchPredicate(threshold));
            candidates.erase(new_end, candidates.end());
        }
        matches.insert(matches.end(), candidates.begin(), candidates.end());
    }
}

int Detector::addTemplate(const Mat source, const std::string &class_id,
                          const Mat &object_mask, int num_features)
{
    std::vector<TemplatePyramid> &template_pyramids = class_templates[class_id];
    int template_id = static_cast<int>(template_pyramids.size());

    TemplatePyramid tp;
    tp.resize(pyramid_levels);

    {
        // Extract a template at each pyramid level
        Ptr<ColorGradientPyramid> qp = modality->process(source, object_mask);

        if(num_features > 0)
        qp->num_features = num_features;

        for (int l = 0; l < pyramid_levels; ++l)
        {
            /// @todo Could do mask subsampling here instead of in pyrDown()
            if (l > 0)
                qp->pyrDown();

            bool success = qp->extractTemplate(tp[l]);
            if (!success)
                return -1;
        }
    }

    //    Rect bb =
    cropTemplates(tp);

    /// @todo Can probably avoid a copy of tp here with swap
    template_pyramids.push_back(tp);
    return template_id;
}
const std::vector<Template> &Detector::getTemplates(const std::string &class_id, int template_id) const
{
    TemplatesMap::const_iterator i = class_templates.find(class_id);
    CV_Assert(i != class_templates.end());
    CV_Assert(i->second.size() > size_t(template_id));
    return i->second[template_id];
}

int Detector::numTemplates() const
{
    int ret = 0;
    TemplatesMap::const_iterator i = class_templates.begin(), iend = class_templates.end();
    for (; i != iend; ++i)
        ret += static_cast<int>(i->second.size());
    return ret;
}

int Detector::numTemplates(const std::string &class_id) const
{
    TemplatesMap::const_iterator i = class_templates.find(class_id);
    if (i == class_templates.end())
        return 0;
    return static_cast<int>(i->second.size());
}

std::vector<std::string> Detector::classIds() const
{
    std::vector<std::string> ids;
    TemplatesMap::const_iterator i = class_templates.begin(), iend = class_templates.end();
    for (; i != iend; ++i)
    {
        ids.push_back(i->first);
    }

    return ids;
}

void Detector::read(const FileNode &fn)
{
    class_templates.clear();
    pyramid_levels = fn["pyramid_levels"];
    fn["T"] >> T_at_level;

    modality = makePtr<ColorGradient>();
}

void Detector::write(FileStorage &fs) const
{
    fs << "pyramid_levels" << pyramid_levels;
    fs << "T" << T_at_level;

    modality->write(fs);
}

std::string Detector::readClass(const FileNode &fn, const std::string &class_id_override)
{
    // Detector should not already have this class
    String class_id;
    if (class_id_override.empty())
    {
        String class_id_tmp = fn["class_id"];
        CV_Assert(class_templates.find(class_id_tmp) == class_templates.end());
        class_id = class_id_tmp;
    }
    else
    {
        class_id = class_id_override;
    }

    TemplatesMap::value_type v(class_id, std::vector<TemplatePyramid>());
    std::vector<TemplatePyramid> &tps = v.second;
    int expected_id = 0;

    FileNode tps_fn = fn["template_pyramids"];
    tps.resize(tps_fn.size());
    FileNodeIterator tps_it = tps_fn.begin(), tps_it_end = tps_fn.end();
    for (; tps_it != tps_it_end; ++tps_it, ++expected_id)
    {
        int template_id = (*tps_it)["template_id"];
        CV_Assert(template_id == expected_id);
        FileNode templates_fn = (*tps_it)["templates"];
        tps[template_id].resize(templates_fn.size());

        FileNodeIterator templ_it = templates_fn.begin(), templ_it_end = templates_fn.end();
        int idx = 0;
        for (; templ_it != templ_it_end; ++templ_it)
        {
            tps[template_id][idx++].read(*templ_it);
        }
    }

    class_templates.insert(v);
    return class_id;
}

void Detector::writeClass(const std::string &class_id, FileStorage &fs) const
{
    TemplatesMap::const_iterator it = class_templates.find(class_id);
    CV_Assert(it != class_templates.end());
    const std::vector<TemplatePyramid> &tps = it->second;

    fs << "class_id" << it->first;
    fs << "pyramid_levels" << pyramid_levels;
    fs << "template_pyramids"
       << "[";
    for (size_t i = 0; i < tps.size(); ++i)
    {
        const TemplatePyramid &tp = tps[i];
        fs << "{";
        fs << "template_id" << int(i); //TODO is this cast correct? won't be good if rolls over...
        fs << "templates"
           << "[";
        for (size_t j = 0; j < tp.size(); ++j)
        {
            fs << "{";
            tp[j].write(fs);
            fs << "}"; // current template
        }
        fs << "]"; // templates
        fs << "}"; // current pyramid
    }
    fs << "]"; // pyramids
}

void Detector::readClasses(const std::vector<std::string> &class_ids,
                           const std::string &format)
{
    for (size_t i = 0; i < class_ids.size(); ++i)
    {
        const String &class_id = class_ids[i];
        String filename = cv::format(format.c_str(), class_id.c_str());
        FileStorage fs(filename, FileStorage::READ);
        readClass(fs.root());
    }
}

void Detector::writeClasses(const std::string &format) const
{
    TemplatesMap::const_iterator it = class_templates.begin(), it_end = class_templates.end();
    for (; it != it_end; ++it)
    {
        const String &class_id = it->first;
        String filename = cv::format(format.c_str(), class_id.c_str());
        FileStorage fs(filename, FileStorage::WRITE);
        writeClass(class_id, fs);
    }
}

} // namespace line2Dup
