#include "line2Dup.h"
#include <iostream>

using namespace std;
using namespace cv;

namespace line2Dup
{
/**
 * \brief Get the label [0,8) of the single bit set in quantized.
 */
static inline int getLabel(int quantized)
{
    switch (quantized)
    {
    case 1:
        return 0;
    case 2:
        return 1;
    case 4:
        return 2;
    case 8:
        return 3;
    case 16:
        return 4;
    case 32:
        return 5;
    case 64:
        return 6;
    case 128:
        return 7;
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

    bool first_select = true;

    while(true)
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

        if (++i == (int)candidates.size()){
            bool num_ok = features.size() >= num_features;

            if(first_select){
                if(num_ok){
                    features.clear(); // we don't want too many first time
                    i = 0;
                    distance += 1.0f;
                    distance_sq = distance * distance;
                    continue;
                }else{
                    first_select = false;
                }
            }

            // Start back at beginning, and relax required distance
            i = 0;
            distance -= 1.0f;
            distance_sq = distance * distance;
            if (num_ok || distance < 3){
                break;
            }
        }
    }
    if (features.size() >= num_features)
    {
        return true;
    }
    else
    {
        std::cout << "this templ has no enough features, but we let it go" << std::endl;
        return true;
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
    Mat_<unsigned char> quantized_unfiltered;
    angle.convertTo(quantized_unfiltered, CV_8U, 16.0 / 360.0);

    // Zero out top and bottom rows
    /// @todo is this necessary, or even correct?
    memset(quantized_unfiltered.ptr(), 0, quantized_unfiltered.cols);
    memset(quantized_unfiltered.ptr(quantized_unfiltered.rows - 1), 0, quantized_unfiltered.cols);
    // Zero out first and last columns
    for (int r = 0; r < quantized_unfiltered.rows; ++r)
    {
        quantized_unfiltered(r, 0) = 0;
        quantized_unfiltered(r, quantized_unfiltered.cols - 1) = 0;
    }

    // Mask 16 buckets into 8 quantized orientations
    for (int r = 1; r < angle.rows - 1; ++r)
    {
        uchar *quant_r = quantized_unfiltered.ptr<uchar>(r);
        for (int c = 1; c < angle.cols - 1; ++c)
        {
            quant_r[c] &= 7;
        }
    }

    // Filter the raw quantized image. Only accept pixels where the magnitude is above some
    // threshold, and there is local agreement on the quantization.
    quantized_angle = Mat::zeros(angle.size(), CV_8U);
    for (int r = 1; r < angle.rows - 1; ++r)
    {
        float *mag_r = magnitude.ptr<float>(r);

        for (int c = 1; c < angle.cols - 1; ++c)
        {
            if (mag_r[c] > threshold)
            {
                // Compute histogram of quantized bins in 3x3 patch around pixel
                int histogram[8] = {0, 0, 0, 0, 0, 0, 0, 0};

                uchar *patch3x3_row = &quantized_unfiltered(r - 1, c - 1);
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
                for (int i = 0; i < 8; ++i)
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
                    quantized_angle.at<uchar>(r, c) = uchar(1 << index);
            }
        }
    }
}

static void quantizedOrientations(const Mat &src, Mat &magnitude,
                                  Mat &angle, float threshold)
{
    Mat smoothed;
    // Compute horizontal and vertical image derivatives on all color channels separately
    static const int KERNEL_SIZE = 7;
    // For some reason cvSmooth/cv::GaussianBlur, cvSobel/cv::Sobel have different defaults for border handling...
    GaussianBlur(src, smoothed, Size(KERNEL_SIZE, KERNEL_SIZE), 0, 0, BORDER_REPLICATE);

    if(src.channels() == 1){
        Mat sobel_dx, sobel_dy, sobel_ag;
        Sobel(smoothed, sobel_dx, CV_32F, 1, 0, 3, 1.0, 0.0, BORDER_REPLICATE);
        Sobel(smoothed, sobel_dy, CV_32F, 0, 1, 3, 1.0, 0.0, BORDER_REPLICATE);
        magnitude = sobel_dx.mul(sobel_dx) + sobel_dy.mul(sobel_dy);
        phase(sobel_dx, sobel_dy, sobel_ag, true);
        hysteresisGradient(magnitude, angle, sobel_ag, threshold * threshold);

    }else{

        magnitude.create(src.size(), CV_32F);

        // Allocate temporary buffers
        Size size = src.size();
        Mat sobel_3dx;              // per-channel horizontal derivative
        Mat sobel_3dy;              // per-channel vertical derivative
        Mat sobel_dx(size, CV_32F); // maximum horizontal derivative
        Mat sobel_dy(size, CV_32F); // maximum vertical derivative
        Mat sobel_ag;               // final gradient orientation (unquantized)

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
    dst = Mat::zeros(angle.size(), CV_8U);
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

                if (score > threshold_sq && angle.at<uchar>(r, c) > 0)
                {
                    candidates.push_back(Candidate(c, r, getLabel(angle.at<uchar>(r, c)), score));
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
    : weak_threshold(30.0f),
      num_features(63),
      strong_threshold(60.0f)
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

static void orUnaligned8u(const uchar *src, const int src_stride,
                          uchar *dst, const int dst_stride,
                          const int width, const int height)
{
    for (int r = 0; r < height; ++r)
    {
        int c = 0;

        // not aligned, which will happen because we move 1 bytes a time for spreading
        while (reinterpret_cast<unsigned long long>(src + c) % 16 != 0) {
            dst[c] |= src[c];
            c++;
        }

        // avoid out of bound when can't divid
        // note: can't use c<width !!!
        for (; c <= width-mipp::N<uint8_t>(); c+=mipp::N<uint8_t>()){
            mipp::Reg<uint8_t> src_v((uint8_t*)src + c);
            mipp::Reg<uint8_t> dst_v((uint8_t*)dst + c);

            mipp::Reg<uint8_t> res_v = mipp::orb(src_v, dst_v);
            res_v.store((uint8_t*)dst + c);
        }

        for(; c<width; c++)
            dst[c] |= src[c];

        // Advance to next row
        src += src_stride;
        dst += dst_stride;
    }
}

static void spread(const Mat &src, Mat &dst, int T)
{
    // Allocate and zero-initialize spread (OR'ed) image
    dst = Mat::zeros(src.size(), CV_8U);

    // Fill in spread gradient image (section 2.3)
    for (int r = 0; r < T; ++r)
    {
        for (int c = 0; c < T; ++c)
        {
            orUnaligned8u(&src.at<unsigned char>(r, c), static_cast<const int>(src.step1()), dst.ptr(),
                          static_cast<const int>(dst.step1()), src.cols - c, src.rows - r);
        }
    }
}

// 1,2-->0 3-->1
CV_DECL_ALIGNED(16)
static const unsigned char SIMILARITY_LUT[256] = {0, 4, 1, 4, 0, 4, 1, 4, 0, 4, 1, 4, 0, 4, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 4, 4, 1, 1, 4, 4, 0, 1, 4, 4, 1, 1, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 4, 4, 4, 4, 1, 1, 1, 1, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 4, 1, 4, 0, 4, 1, 4, 0, 4, 1, 4, 0, 4, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 4, 1, 1, 4, 4, 0, 1, 4, 4, 1, 1, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 4, 4, 4, 4, 1, 1, 1, 1, 4, 4, 4, 4, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4};

static void computeResponseMaps(const Mat &src, std::vector<Mat> &response_maps)
{
    CV_Assert((src.rows * src.cols) % 16 == 0);

    // Allocate response maps
    response_maps.resize(8);
    for (int i = 0; i < 8; ++i)
        response_maps[i].create(src.size(), CV_8U);

    Mat lsb4(src.size(), CV_8U);
    Mat msb4(src.size(), CV_8U);

    for (int r = 0; r < src.rows; ++r)
    {
        const uchar *src_r = src.ptr(r);
        uchar *lsb4_r = lsb4.ptr(r);
        uchar *msb4_r = msb4.ptr(r);

        for (int c = 0; c < src.cols; ++c)
        {
            // Least significant 4 bits of spread image pixel
            lsb4_r[c] = src_r[c] & 15;
            // Most significant 4 bits, right-shifted to be in [0, 16)
            msb4_r[c] = (src_r[c] & 240) >> 4;
        }
    }

    {
        uchar *lsb4_data = lsb4.ptr<uchar>();
        uchar *msb4_data = msb4.ptr<uchar>();

        bool no_max = true;
        bool no_shuff = true;

#ifdef has_max_int8_t
        no_max = false;
#endif

#ifdef has_shuff_int8_t
        no_shuff = false;
#endif
        // LUT is designed for 128 bits SIMD, so quite triky for others

        // For each of the 8 quantized orientations...
        for (int ori = 0; ori < 8; ++ori){
            uchar *map_data = response_maps[ori].ptr<uchar>();
            const uchar *lut_low = SIMILARITY_LUT + 32 * ori;

            if(mipp::N<uint8_t>() == 1 || no_max || no_shuff){ // no SIMD
                for (int i = 0; i < src.rows * src.cols; ++i)
                    map_data[i] = std::max(lut_low[lsb4_data[i]], lut_low[msb4_data[i] + 16]);
            }
            else if(mipp::N<uint8_t>() == 16){ // 128 SIMD, no add base

                const uchar *lut_low = SIMILARITY_LUT + 32 * ori;
                mipp::Reg<uint8_t> lut_low_v((uint8_t*)lut_low);
                mipp::Reg<uint8_t> lut_high_v((uint8_t*)lut_low + 16);

                for (int i = 0; i < src.rows * src.cols; i += mipp::N<uint8_t>()){
                    mipp::Reg<uint8_t> low_mask((uint8_t*)lsb4_data + i);
                    mipp::Reg<uint8_t> high_mask((uint8_t*)msb4_data + i);

                    mipp::Reg<uint8_t> low_res = mipp::shuff(lut_low_v, low_mask);
                    mipp::Reg<uint8_t> high_res = mipp::shuff(lut_high_v, high_mask);

                    mipp::Reg<uint8_t> result = mipp::max(low_res, high_res);
                    result.store((uint8_t*)map_data + i);
                }
            }
            else if(mipp::N<uint8_t>() == 16 || mipp::N<uint8_t>() == 32
                    || mipp::N<uint8_t>() == 64){ //128 256 512 SIMD
                CV_Assert((src.rows * src.cols) % mipp::N<uint8_t>() == 0);

                uint8_t lut_temp[mipp::N<uint8_t>()] = {0};

                for(int slice=0; slice<mipp::N<uint8_t>()/16; slice++){
                    std::copy_n(lut_low, 16, lut_temp+slice*16);
                }
                mipp::Reg<uint8_t> lut_low_v(lut_temp);

                uint8_t base_add_array[mipp::N<uint8_t>()] = {0};
                for(uint8_t slice=0; slice<mipp::N<uint8_t>(); slice+=16){
                    std::copy_n(lut_low+16, 16, lut_temp+slice);
                    std::fill_n(base_add_array+slice, 16, slice);
                }
                mipp::Reg<uint8_t> base_add(base_add_array);
                mipp::Reg<uint8_t> lut_high_v(lut_temp);

                for (int i = 0; i < src.rows * src.cols; i += mipp::N<uint8_t>()){
                    mipp::Reg<uint8_t> mask_low_v((uint8_t*)lsb4_data+i);
                    mipp::Reg<uint8_t> mask_high_v((uint8_t*)msb4_data+i);

                    mask_low_v += base_add;
                    mask_high_v += base_add;

                    mipp::Reg<uint8_t> shuff_low_result = mipp::shuff(lut_low_v, mask_low_v);
                    mipp::Reg<uint8_t> shuff_high_result = mipp::shuff(lut_high_v, mask_high_v);

                    mipp::Reg<uint8_t> result = mipp::max(shuff_low_result, shuff_high_result);
                    result.store((uint8_t*)map_data + i);
                }
            }
            else{
                for (int i = 0; i < src.rows * src.cols; ++i)
                    map_data[i] = std::max(lut_low[lsb4_data[i]], lut_low[msb4_data[i] + 16]);
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
    // we only have one modality, so 8192*2, due to mipp, back to 8192
    CV_Assert(templ.features.size() < 8192);

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
    mipp::Reg<uint8_t> zero_v(uint8_t(0));

    for (int i = 0; i < (int)templ.features.size(); ++i)
    {

        Feature f = templ.features[i];

        if (f.x < 0 || f.x >= size.width || f.y < 0 || f.y >= size.height)
            continue;
        const uchar *lm_ptr = accessLinearMemory(linear_memories, f, T, W);

        int j = 0;

        // *2 to avoid int8 read out of range
        for(; j <= template_positions -mipp::N<int16_t>()*2; j+=mipp::N<int16_t>()){
            mipp::Reg<uint8_t> src8_v((uint8_t*)lm_ptr + j);

            // uchar to short, once for N bytes
            mipp::Reg<int16_t> src16_v(mipp::interleavelo(src8_v, zero_v).r);

            mipp::Reg<int16_t> dst_v((int16_t*)dst_ptr + j);

            mipp::Reg<int16_t> res_v = src16_v + dst_v;
            res_v.store((int16_t*)dst_ptr + j);
        }

        for(; j<template_positions; j++)
            dst_ptr[j] += short(lm_ptr[j]);
    }
}

static void similarityLocal(const std::vector<Mat> &linear_memories, const Template &templ,
                            Mat &dst, Size size, int T, Point center)
{
    CV_Assert(templ.features.size() < 8192);

    int W = size.width / T;
    dst = Mat::zeros(16, 16, CV_16U);

    int offset_x = (center.x / T - 8) * T;
    int offset_y = (center.y / T - 8) * T;
    mipp::Reg<uint8_t> zero_v = uint8_t(0);

    for (int i = 0; i < (int)templ.features.size(); ++i)
    {
        Feature f = templ.features[i];
        f.x += offset_x;
        f.y += offset_y;
        // Discard feature if out of bounds, possibly due to applying the offset
        if (f.x < 0 || f.y < 0 || f.x >= size.width || f.y >= size.height)
            continue;

        const uchar *lm_ptr = accessLinearMemory(linear_memories, f, T, W);
        {
            short *dst_ptr = dst.ptr<short>();

            if(mipp::N<uint8_t>() > 32){ //512 bits SIMD
                for (int row = 0; row < 16; row += mipp::N<int16_t>()/16){
                    mipp::Reg<int16_t> dst_v((int16_t*)dst_ptr + row*16);

                    // load lm_ptr, 16 bytes once, for half
                    uint8_t local_v[mipp::N<uint8_t>()] = {0};
                    for(int slice=0; slice<mipp::N<uint8_t>()/16/2; slice++){
                        std::copy_n(lm_ptr, 16, &local_v[16*slice]);
                        lm_ptr += W;
                    }
                    mipp::Reg<uint8_t> src8_v(local_v);
                    // uchar to short, once for N bytes
                    mipp::Reg<int16_t> src16_v(mipp::interleavelo(src8_v, zero_v).r);

                    mipp::Reg<int16_t> res_v = src16_v + dst_v;
                    res_v.store((int16_t*)dst_ptr);

                    dst_ptr += mipp::N<int16_t>();
                }
            }else{ // 256 128 or no SIMD
                for (int row = 0; row < 16; ++row){
                    for(int col=0; col<16; col+=mipp::N<int16_t>()){
                        mipp::Reg<uint8_t> src8_v((uint8_t*)lm_ptr + col);

                        // uchar to short, once for N bytes
                        mipp::Reg<int16_t> src16_v(mipp::interleavelo(src8_v, zero_v).r);

                        mipp::Reg<int16_t> dst_v((int16_t*)dst_ptr + col);
                        mipp::Reg<int16_t> res_v = src16_v + dst_v;
                        res_v.store((int16_t*)dst_ptr + col);
                    }
                    dst_ptr += 16;
                    lm_ptr += W;
                }
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
    CV_Assert(templ.features.size() < 64);
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

        for(; j <= template_positions -mipp::N<uint8_t>(); j+=mipp::N<uint8_t>()){
            mipp::Reg<uint8_t> src_v((uint8_t*)lm_ptr + j);
            mipp::Reg<uint8_t> dst_v((uint8_t*)dst_ptr + j);

            mipp::Reg<uint8_t> res_v = src_v + dst_v;
            res_v.store((uint8_t*)dst_ptr + j);
        }

        for(; j<template_positions; j++)
            dst_ptr[j] += lm_ptr[j];
    }
}

static void similarityLocal_64(const std::vector<Mat> &linear_memories, const Template &templ,
                               Mat &dst, Size size, int T, Point center)
{
    // Similar to whole-image similarity() above. This version takes a position 'center'
    // and computes the energy in the 16x16 patch centered on it.
    CV_Assert(templ.features.size() < 64);

    // Compute the similarity map in a 16x16 patch around center
    int W = size.width / T;
    dst = Mat::zeros(16, 16, CV_8U);

    // Offset each feature point by the requested center. Further adjust to (-8,-8) from the
    // center to get the top-left corner of the 16x16 patch.
    // NOTE: We make the offsets multiples of T to agree with results of the original code.
    int offset_x = (center.x / T - 8) * T;
    int offset_y = (center.y / T - 8) * T;

    for (int i = 0; i < (int)templ.features.size(); ++i)
    {
        Feature f = templ.features[i];
        f.x += offset_x;
        f.y += offset_y;
        // Discard feature if out of bounds, possibly due to applying the offset
        if (f.x < 0 || f.y < 0 || f.x >= size.width || f.y >= size.height)
            continue;

        const uchar *lm_ptr = accessLinearMemory(linear_memories, f, T, W);

        {
            uchar *dst_ptr = dst.ptr<uchar>();

            if(mipp::N<uint8_t>() > 16){ // 256 or 512 bits SIMD
                for (int row = 0; row < 16; row += mipp::N<uint8_t>()/16){
                    mipp::Reg<uint8_t> dst_v((uint8_t*)dst_ptr);

                    // load lm_ptr, 16 bytes once
                    uint8_t local_v[mipp::N<uint8_t>()];
                    for(int slice=0; slice<mipp::N<uint8_t>()/16; slice++){
                        std::copy_n(lm_ptr, 16, &local_v[16*slice]);
                        lm_ptr += W;
                    }
                    mipp::Reg<uint8_t> src_v(local_v);

                    mipp::Reg<uint8_t> res_v = src_v + dst_v;
                    res_v.store((uint8_t*)dst_ptr);

                    dst_ptr += mipp::N<uint8_t>();
                }
            }else{ // 128 or no SIMD
                for (int row = 0; row < 16; ++row){
                    for(int col=0; col<16; col+=mipp::N<uint8_t>()){
                        mipp::Reg<uint8_t> src_v((uint8_t*)lm_ptr + col);
                        mipp::Reg<uint8_t> dst_v((uint8_t*)dst_ptr + col);
                        mipp::Reg<uint8_t> res_v = src_v + dst_v;
                        res_v.store((uint8_t*)dst_ptr + col);
                    }
                    dst_ptr += 16;
                    lm_ptr += W;
                }
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
    T_at_level.push_back(4);
    T_at_level.push_back(8);
}

Detector::Detector(std::vector<int> T)
{
    this->modality = makePtr<ColorGradient>();
    pyramid_levels = T.size();
    T_at_level = T;
}

Detector::Detector(int num_features, std::vector<int> T, float weak_thresh, float strong_threash)
{
    this->modality = makePtr<ColorGradient>(weak_thresh, num_features, strong_threash);
    pyramid_levels = T.size();
    T_at_level = T;
    res_map_mag_thresh = strong_threash;
}

static int gcd(int a, int b){
    if (a == 0)
        return b;
    return gcd(b % a, a);
}
static int lcm(int a, int b){
    return (a*b)/gcd(a, b);
}
static int least_mul_of_Ts(const std::vector<int>& T_at_level){
    assert(T_at_level.size() > 0);
    int cur_res = T_at_level[0];
    for(int i=1; i<T_at_level.size(); i++){
        int cur_v = T_at_level[i] << i;
        cur_res = lcm(cur_v, cur_res);
    }
    return cur_res;
}

std::vector<Match> Detector::match(Mat source, float threshold, const std::vector<string> &class_ids, const Mat mask) const
{
    std::vector<Match> matches;

    // lm_pyramid sizes
    // --------- provided by construction of response map, fusion version now
    LinearMemoryPyramid lm_pyramid(pyramid_levels, std::vector<LinearMemories>(1, LinearMemories(8)));
    std::vector<Size> sizes;

    const bool use_fusion = true;
    if(use_fusion){  // fusion version
        assert(source.channels() == 1 && "only gray img now");
        assert(mask.empty() && "mask not support yet");

        const int tileRows = 32;
        const int tileCols = 256;
        const int num_threads = 1;
        const bool use_hist3x3 = true;

        const int32_t mag_thresh_l2 = int32_t(res_map_mag_thresh*res_map_mag_thresh);

        const int lcm_Ts = least_mul_of_Ts(T_at_level);
        const int biggest_imgRows = source.rows/lcm_Ts*lcm_Ts;
        const int biggest_imgCols = source.cols/lcm_Ts*lcm_Ts;

        // gaussian coff quantization
        const int gauss_size = 5;
        const int gauss_quant_bit = 8;
        cv::Mat double_gauss = cv::getGaussianKernel(gauss_size, 0, CV_64F);
        int32_t gauss_knl_uint32[gauss_size] = {0};
        for(int i=0; i<gauss_size; i++){
            gauss_knl_uint32[i] = int32_t(double_gauss.at<double>(i, 0) * (1<<gauss_quant_bit));
        }

        cv::Mat pyr_src;
        for(int cur_l = 0; cur_l<T_at_level.size(); cur_l++){
            const bool need_pyr = cur_l < T_at_level.size() - 1;

            const int imgRows = biggest_imgRows >> cur_l;
            const int imgCols = biggest_imgCols >> cur_l;

            const int cur_T = T_at_level[cur_l];
            assert(cur_T % 2 == 0);

            for(int ori=0; ori<8; ori++){
                lm_pyramid[cur_l][0][ori] = cv::Mat(cur_T*cur_T, imgCols/cur_T*imgRows/cur_T, CV_8U, cv::Scalar(0));
            }

            const int thread_rows_step = imgRows / num_threads;
            sizes.push_back({imgCols, imgRows});

            cv::Mat src;
            if(cur_l == 0) src = source;
            else src = pyr_src;

            if(need_pyr) pyr_src = cv::Mat(imgRows/2, imgCols/2, CV_16U, cv::Scalar(0));

//#pragma omp parallel for num_threads(num_threads)
            for(int thread_i = 0; thread_i < num_threads; thread_i++){
                const int tile_start_rows = thread_i * thread_rows_step;
                const int tile_end_rows = tile_start_rows + thread_rows_step;
                std::vector<line2Dup::FilterNode> nodes;
                // node graph:
                // src -- gx 1x5 -- img0 -- gy 5x1 -- img1 -- sxx 1x3 -- img2 -- sxy 3x1 -- img3 -- mag         1x1 -- img4 -- hist 3x3 -- img5
                //                                         -- syx 1x3 --      -- syy 3x1 --         phase+quant 1x1
                //  if need pyrDown:                  img1 -- pyrDown -- pyr_src
                // img5 -- spread 1x5 -- img6 -- spread 5x1 -- img7 -- response 1x1 -- img8 -- linearize no padding -- img9

                // assign nodes, put it here because all threads have its own buffer
                int cur_n = 0;
                nodes.push_back(FilterNode());
                nodes[cur_n].op_name = "gx 1xgauss_size";
                nodes[cur_n].op_r = 1;
                nodes[cur_n].op_c = gauss_size;
                nodes[cur_n].op_type = CV_32S;
                nodes[cur_n].simd_step = mipp::N<int32_t>();
                {
                    nodes[cur_n].simple_update = [&nodes, cur_n, &src, gauss_knl_uint32, imgCols]
                            (int start_r, int end_r, int start_c, int end_c) {
                        auto &cur_node = nodes[cur_n];
                        // src has no padding, so update c should be padded
                        if(start_c < gauss_size/2) start_c = gauss_size/2;
                        if(end_c >= imgCols - gauss_size/2) end_c = imgCols - gauss_size/2;

                        int c = start_c;
                        for (int r = start_r; r < end_r; r++){
                            c = start_c;
                            int32_t *buf_ptr = cur_node.ptr<int32_t>(r, c);

                            if(src.type() == CV_16U){
                                const int16_t *parent_buf_ptr = &src.at<int16_t>(r, c);
                                for (; c < end_c; c++, buf_ptr++, parent_buf_ptr++){

                                    int32_t local_sum = gauss_knl_uint32[gauss_size/2]*int32_t(*(parent_buf_ptr));
                                    int gauss_knl_idx = gauss_size/2 + 1;
                                    for(int i=1; i<=gauss_size/2; i++, gauss_knl_idx++){
                                        local_sum += (int32_t(*(parent_buf_ptr + i)) +
                                                      int32_t(*(parent_buf_ptr - i))) * gauss_knl_uint32[gauss_knl_idx];
                                    }
                                    *buf_ptr = local_sum;
                                }
                            }else if(src.type() == CV_8U){
                                const uint8_t *parent_buf_ptr = &src.at<uint8_t>(r, c);
                                for (; c < end_c; c++, buf_ptr++, parent_buf_ptr++){

                                    int32_t local_sum = gauss_knl_uint32[gauss_size/2]*int32_t(*(parent_buf_ptr));
                                    int gauss_knl_idx = gauss_size/2 + 1;
                                    for(int i=1; i<=gauss_size/2; i++, gauss_knl_idx++){
                                        local_sum += (int32_t(*(parent_buf_ptr + i)) +
                                                      int32_t(*(parent_buf_ptr - i))) * gauss_knl_uint32[gauss_knl_idx];
                                    }
                                    *buf_ptr = local_sum;
                                }
                            }else{
                                CV_Error(Error::StsBadArg, "Invalid type of src");
                            }
                        }
                        return c;
                    };
                    nodes[cur_n].simd_update = [&nodes, cur_n, &src, imgCols, gauss_knl_uint32]
                            (int start_r, int end_r, int start_c, int end_c) {
                        auto &cur_node = nodes[cur_n];
                        // src has no padding, so update c should be padded
                        if(start_c < gauss_size/2) start_c = gauss_size/2;
                        if(end_c >= imgCols - gauss_size/2) end_c = imgCols - gauss_size/2;

                        mipp::Reg<uint8_t> zero8_v = uint8_t(0);
                        mipp::Reg<int16_t> zero16_v = int16_t(0);
                        int c = start_c;
                        for (int r = start_r; r < end_r; r++){
                            c = start_c;
                            int32_t *buf_ptr = cur_node.ptr<int32_t>(r, c);

                            if(src.type() == CV_16U){
                                const int16_t *parent_buf_ptr = &src.at<int16_t>(r, c);
                                for (; c < end_c; c += cur_node.simd_step,
                                    buf_ptr += cur_node.simd_step, parent_buf_ptr += cur_node.simd_step){
                                    if (c + 2*cur_node.simd_step >= end_c)
                                        break; // simd may excel end_c, but avoid simd out of img, *4 because int32 = 4*8

                                    mipp::Reg<int32_t> gauss_coff0(gauss_knl_uint32[gauss_size/2]);
                                    mipp::Reg<int16_t> src16_v0(parent_buf_ptr);
                                    mipp::Reg<int32_t> src32_v0(mipp::interleavelo(src16_v0, zero16_v).r);
                                    mipp::Reg<int32_t> local_sum = gauss_coff0*src32_v0;
                                    int gauss_knl_idx = gauss_size/2 + 1;
                                    for(int i=1; i<=gauss_size/2; i++, gauss_knl_idx++){
                                        mipp::Reg<int32_t> gauss_coff(gauss_knl_uint32[gauss_knl_idx]);
                                        mipp::Reg<int16_t> src16_v1(parent_buf_ptr + i);
                                        mipp::Reg<int32_t> src32_v1(mipp::interleavelo(src16_v1, zero16_v).r);

                                        mipp::Reg<int16_t> src16_v2(parent_buf_ptr - i);
                                        mipp::Reg<int32_t> src32_v2(mipp::interleavelo(src16_v2, zero16_v).r);
                                        local_sum += gauss_coff * (src32_v1 + src32_v2);
                                    }
                                    local_sum.store(buf_ptr);
                                }
                            }else if(src.type() == CV_8U){
                                const uint8_t *parent_buf_ptr = &src.at<uint8_t>(r, c);
                                for (; c < end_c; c += cur_node.simd_step,
                                    buf_ptr += cur_node.simd_step, parent_buf_ptr += cur_node.simd_step){
                                    if (c + 4*cur_node.simd_step >= end_c)
                                        break; // simd may excel end_c, but avoid simd out of img, *4 because int32 = 4*8

                                    mipp::Reg<int32_t> gauss_coff0(gauss_knl_uint32[gauss_size/2]);
                                    mipp::Reg<uint8_t> src8_v0(parent_buf_ptr);
                                    mipp::Reg<int16_t> src16_v0(mipp::interleavelo(src8_v0, zero8_v).r);
                                    mipp::Reg<int32_t> src32_v0(mipp::interleavelo(src16_v0, zero16_v).r);
                                    mipp::Reg<int32_t> local_sum = gauss_coff0*src32_v0;
                                    int gauss_knl_idx = gauss_size/2 + 1;
                                    for(int i=1; i<=gauss_size/2; i++, gauss_knl_idx++){
                                        mipp::Reg<int32_t> gauss_coff(gauss_knl_uint32[gauss_knl_idx]);
                                        mipp::Reg<uint8_t> src8_v1(parent_buf_ptr + i);
                                        mipp::Reg<int16_t> src16_v1(mipp::interleavelo(src8_v1, zero8_v).r);
                                        mipp::Reg<int32_t> src32_v1(mipp::interleavelo(src16_v1, zero16_v).r);

                                        mipp::Reg<uint8_t> src8_v2(parent_buf_ptr - i);
                                        mipp::Reg<int16_t> src16_v2(mipp::interleavelo(src8_v2, zero8_v).r);
                                        mipp::Reg<int32_t> src32_v2(mipp::interleavelo(src16_v2, zero16_v).r);
                                        local_sum += gauss_coff * (src32_v1 + src32_v2);
                                    }
                                    local_sum.store(buf_ptr);
                                }
                            }else{
                                CV_Error(Error::StsBadArg, "Invalid type of src");
                            }
                        }
                        if (c < end_c)
                            return cur_node.simple_update(start_r, end_r, c, end_c);
                        return c;
                    };
                }

                cur_n++; // 1
                nodes.push_back(FilterNode());
                nodes[cur_n].op_name = "gy gauss_sizex1";
                nodes[cur_n].op_r = gauss_size;
                nodes[cur_n].op_c = 1;
                nodes[cur_n].parent = cur_n - 1;
                nodes[cur_n].simd_step = mipp::N<int32_t>();  // type is 16U, but step is 32
                {
                    nodes[cur_n].simple_update = [&nodes, cur_n, gauss_knl_uint32, need_pyr, &pyr_src, tile_end_rows]
                            (int start_r, int end_r, int start_c, int end_c){
                        auto &cur_node = nodes[cur_n];
                        auto &parent_node = nodes[cur_node.parent];
                        int c = start_c;
                        for(int r = start_r; r < end_r; r++){
                            c = start_c;
                            int16_t *buf_ptr = cur_node.ptr<int16_t>(r, c);
                            int16_t *pyr_src_ptr = &pyr_src.at<int16_t>(r/2, c/2);

                            int32_t* parent_buf_ptr[gauss_size];
                            int32_t** parent_buf_center = parent_buf_ptr + gauss_size/2;
                            parent_buf_center[0] = parent_node.ptr<int32_t>(r, c);
                            for(int i=1; i<=gauss_size/2; i++){
                                parent_buf_center[i] = parent_node.ptr<int32_t>(r+i, c);
                                parent_buf_center[-i] = parent_node.ptr<int32_t>(r-i, c);
                            }
                            bool is_even_row = r%2 == 0;
                            for (; c < end_c; c++, buf_ptr++){
                                int32_t local_sum = gauss_knl_uint32[gauss_size/2] * (*(parent_buf_center[0]));
                                int gauss_knl_idx = gauss_size/2 + 1;
                                for(int i=1; i<=gauss_size/2; i++, gauss_knl_idx++){
                                    local_sum += gauss_knl_uint32[gauss_knl_idx] *
                                            (*(parent_buf_center[i]) + *(parent_buf_center[-i]));
                                }
                                *buf_ptr = int16_t(local_sum >> (2*gauss_quant_bit));

                                if(need_pyr && is_even_row && r<tile_end_rows){  // avoid out of current thread's rows
                                    if(c%2==0){
                                        *pyr_src_ptr = *buf_ptr;
                                        pyr_src_ptr++;
                                    }
                                }

                                for(int i=0; i<gauss_size; i++){
                                    parent_buf_ptr[i]++;
                                }
                            }
                        }
                        return c;
                    };
                    nodes[cur_n].simd_update = [&nodes, cur_n, imgCols, gauss_knl_uint32, need_pyr, &pyr_src, tile_end_rows]
                            (int start_r, int end_r, int start_c, int end_c) {
                        auto &cur_node = nodes[cur_n];
                        auto &parent_node = nodes[cur_node.parent];
                        int c = start_c;

                        mipp::Reg<int16_t> zero16_v = int16_t(0);
                        mipp::Reg<int32_t> zero32_v = int32_t(0);
                        for (int r = start_r; r < end_r; r++){
                            c = start_c;
                            int16_t *buf_ptr = cur_node.ptr<int16_t>(r, c);

                            int32_t* parent_buf_ptr[gauss_size];
                            int32_t** parent_buf_center = parent_buf_ptr + gauss_size/2;
                            parent_buf_ptr[gauss_size/2] = parent_node.ptr<int32_t>(r, c);
                            for(int i=1; i<=gauss_size/2; i++){
                                parent_buf_center[i] = parent_node.ptr<int32_t>(r+i, c);
                                parent_buf_center[-i] = parent_node.ptr<int32_t>(r-i, c);
                            }

                            bool is_even_row = r%2 == 0;
                            int is_not_even_col = int(c%2 == 1);
                            int16_t* pyr_src_ptr = &pyr_src.at<int16_t>(r/2, c/2);

                            // once for two simd step, because we need to fill the pyrdown lane
                            for (; c < end_c; c += 2*cur_node.simd_step, buf_ptr += 2*cur_node.simd_step,
                                  pyr_src_ptr += cur_node.simd_step){
                                if (c + 4*cur_node.simd_step >= end_c)
                                    break; // simd may excel end_c, but avoid simd out of img

                                int c_stride = 0;
                                for(int c_step = 0; c_step < 2; c_step++, c_stride+=cur_node.simd_step){

                                    mipp::Reg<int32_t> gauss_coff0(gauss_knl_uint32[gauss_size/2]);
                                    mipp::Reg<int32_t> src32_v0(parent_buf_center[0]);
                                    mipp::Reg<int32_t> local_sum = gauss_coff0 * src32_v0;
                                    int gauss_knl_idx = gauss_size/2 + 1;
                                    for(int i=1; i<=gauss_size/2; i++, gauss_knl_idx++){
                                        mipp::Reg<int32_t> gauss_coff(gauss_knl_uint32[gauss_knl_idx]);
                                        mipp::Reg<int32_t> src32_v1(parent_buf_center[i]);
                                        mipp::Reg<int32_t> src32_v2(parent_buf_center[-i]);

                                        local_sum += gauss_coff * (src32_v1 + src32_v2);
                                    }
                                    local_sum >>= (2*gauss_quant_bit);

                                    // two lanes may save one pack?
                                    mipp::Reg<int16_t> local_sum_int16 = mipp::pack<int32_t,int16_t>(local_sum, zero32_v);
                                    local_sum_int16.store(buf_ptr + c_stride);

                                    for(int i=0; i<gauss_size; i++){
                                        parent_buf_ptr[i]+=cur_node.simd_step;
                                    }
                                }
                                if(need_pyr && r<tile_end_rows && is_even_row){

                                    // simd version may have too many overhead
                                    const bool simd_pyr_version = false;
                                    if(simd_pyr_version){
                                        mipp::Reg<int32_t> pyrdown_src((int32_t*)buf_ptr);
                                        if(is_not_even_col){
                                            pyrdown_src >>= 16; // move to low part
                                        }else{
                                            pyrdown_src <<= 16; // make high part zero
                                            pyrdown_src >>= 16; // move to low part
                                        }

                                        // two lanes may save one pack?
                                        mipp::Reg<int16_t> result = mipp::pack<int32_t, int16_t>(pyrdown_src, zero32_v);
                                        result.store(pyr_src_ptr+is_not_even_col);
                                    }else{
                                        int i=0;
                                        int j=0;
                                        for(; i<2*cur_node.simd_step; i+=2, j++){
                                            *(pyr_src_ptr + j + is_not_even_col) = *(buf_ptr + is_not_even_col + i);
                                        }
                                    }
                                }
                            }
                        }
                        if (c < end_c)
                            return cur_node.simple_update(start_r, end_r, c, end_c);
                        return c;
                    };
                }

                cur_n++; // 2
                nodes.push_back(FilterNode());
                nodes[cur_n].op_name = "sxx syx 1x3";
                nodes[cur_n].op_r = 1;
                nodes[cur_n].op_c = 3;
                nodes[cur_n].op_type = CV_16S;
                nodes[cur_n].parent = cur_n - 1;
                nodes[cur_n].num_buf = 2;
                {
                    nodes[cur_n].simple_update = [&nodes, cur_n](int start_r, int end_r, int start_c, int end_c) {
                        auto &cur_node = nodes[cur_n];
                        auto &parent_node = nodes[cur_node.parent];

                        int c = start_c;
                        for (int r = start_r; r < end_r; r++){
                            c = start_c;
                            int16_t *buf_ptr_0 = cur_node.ptr<int16_t>(r, c, 0);
                            int16_t *buf_ptr_1 = cur_node.ptr<int16_t>(r, c, 1);
                            int16_t *parent_buf_ptr = parent_node.ptr<int16_t>(r, c);
                            for (; c < end_c; c++, buf_ptr_0++, buf_ptr_1++, parent_buf_ptr++){
                                // sxx  -1 0 1
                                *buf_ptr_0 = -*(parent_buf_ptr-1) + *(parent_buf_ptr+1);
                                // syx   1 2 1
                                *buf_ptr_1 = *(parent_buf_ptr-1) + 2*(*parent_buf_ptr) + *(parent_buf_ptr+1);
                            }
                        }
                        return c;
                    };
                    nodes[cur_n].simd_update = [&nodes, cur_n, imgCols](int start_r, int end_r, int start_c, int end_c) {
                        auto &cur_node = nodes[cur_n];
                        auto &parent_node = nodes[cur_node.parent];

                        int c = start_c;
                        for (int r = start_r; r < end_r; r++){
                            c = start_c;
                            int16_t *buf_ptr_0 = cur_node.ptr<int16_t>(r, c, 0);
                            int16_t *buf_ptr_1 = cur_node.ptr<int16_t>(r, c, 1);
                            int16_t *parent_buf_ptr = parent_node.ptr<int16_t>(r, c);

                            for (; c < end_c; c += cur_node.simd_step, buf_ptr_0 += cur_node.simd_step,
                                 buf_ptr_1 += cur_node.simd_step, parent_buf_ptr += cur_node.simd_step){
                                if (c + cur_node.simd_step >= end_c)
                                    break; // simd may excel end_c, but avoid simd out of img

                                mipp::Reg<int16_t> p0(parent_buf_ptr-1);
                                mipp::Reg<int16_t> p1(parent_buf_ptr);
                                mipp::Reg<int16_t> p2(parent_buf_ptr+1);

                                mipp::Reg<int16_t> sxx = p2 - p0;

                                // bit shift is too dangorous for signed value
                                // mipp::Reg<int16_t> sxy = p2 + p0 + (p1 << 1);

                                mipp::Reg<int16_t> two_int16 = int16_t(2);
                                mipp::Reg<int16_t> syx = p2 + p0 + (p1 * two_int16);

                                sxx.store(buf_ptr_0);
                                syx.store(buf_ptr_1);
                            }
                        }

                        if (c < end_c)
                            return cur_node.simple_update(start_r, end_r, c, end_c);

                        return c;
                    };
                }

                cur_n++; // 3
                nodes.push_back(FilterNode());
                nodes[cur_n].op_name = "sxy syy 3x1";
                nodes[cur_n].op_r = 3;
                nodes[cur_n].op_c = 1;
                nodes[cur_n].op_type = CV_16S;
                nodes[cur_n].parent = cur_n - 1;
                nodes[cur_n].num_buf = 2;
                {
                    nodes[cur_n].simple_update = [&nodes, cur_n](int start_r, int end_r, int start_c, int end_c) {
                        auto &cur_node = nodes[cur_n];
                        auto &parent_node = nodes[cur_node.parent];
                        int c = start_c;
                        for (int r = start_r; r < end_r; r++){
                            c = start_c;
                            int16_t *buf_ptr_0 = cur_node.ptr<int16_t>(r, c, 0);
                            int16_t *buf_ptr_1 = cur_node.ptr<int16_t>(r, c, 1);
                            int16_t *parent_buf_ptr_0 = parent_node.ptr<int16_t>(r, c, 0);

                            int16_t *parent_buf_ptr_0_ = parent_node.ptr<int16_t>(r-1, c, 0);
                            int16_t *parent_buf_ptr_1_ = parent_node.ptr<int16_t>(r-1, c, 1);

                            int16_t *parent_buf_ptr_0__ = parent_node.ptr<int16_t>(r+1, c, 0);
                            int16_t *parent_buf_ptr_1__ = parent_node.ptr<int16_t>(r+1, c, 1);

                            for(; c < end_c; c++, buf_ptr_0++, buf_ptr_1++, parent_buf_ptr_0++,
                            parent_buf_ptr_0_++, parent_buf_ptr_1_++, parent_buf_ptr_0__++, parent_buf_ptr_1__++){
                                // sxy  1 2 1
                                *buf_ptr_0 = *(parent_buf_ptr_0_) +
                                        2*(*parent_buf_ptr_0) + *(parent_buf_ptr_0__);

                                // syy  -1 0 1
                                *buf_ptr_1 = -*(parent_buf_ptr_1_) + *(parent_buf_ptr_1__);
                            }
                        }
                        return c;
                    };
                    nodes[cur_n].simd_update = [&nodes, cur_n, imgCols](int start_r, int end_r, int start_c, int end_c) {
                        auto &cur_node = nodes[cur_n];
                        auto &parent_node = nodes[cur_node.parent];
                        int c = start_c;
                        for (int r = start_r; r < end_r; r++){
                            c = start_c;
                            int16_t *buf_ptr_0 = cur_node.ptr<int16_t>(r, c, 0);
                            int16_t *buf_ptr_1 = cur_node.ptr<int16_t>(r, c, 1);
                            int16_t *parent_buf_ptr_0 = parent_node.ptr<int16_t>(r, c, 0);

                            int16_t *parent_buf_ptr_0_ = parent_node.ptr<int16_t>(r-1, c, 0);
                            int16_t *parent_buf_ptr_1_ = parent_node.ptr<int16_t>(r-1, c, 1);

                            int16_t *parent_buf_ptr_0__ = parent_node.ptr<int16_t>(r+1, c, 0);
                            int16_t *parent_buf_ptr_1__ = parent_node.ptr<int16_t>(r+1, c, 1);

                            for (; c < end_c; c += cur_node.simd_step, buf_ptr_0 += cur_node.simd_step, buf_ptr_1 += cur_node.simd_step,
                                 parent_buf_ptr_0 += cur_node.simd_step,
                                 parent_buf_ptr_0_ += cur_node.simd_step, parent_buf_ptr_1_ += cur_node.simd_step,
                                 parent_buf_ptr_0__ += cur_node.simd_step, parent_buf_ptr_1__ += cur_node.simd_step){
                                if (c + cur_node.simd_step >= end_c)
                                    break; // simd may excel end_c, but avoid simd out of img
                                {
                                    mipp::Reg<int16_t> p0(parent_buf_ptr_0_);
                                    mipp::Reg<int16_t> p1(parent_buf_ptr_0);
                                    mipp::Reg<int16_t> p2(parent_buf_ptr_0__);

                                    mipp::Reg<int16_t> two_int16 = int16_t(2);
                                    mipp::Reg<int16_t> sxy = p2 + p0 + (p1 * two_int16);
                                    sxy.store(buf_ptr_0);
                                }
                                {
                                    mipp::Reg<int16_t> p0(parent_buf_ptr_1_);
                                    mipp::Reg<int16_t> p2(parent_buf_ptr_1__);

                                    mipp::Reg<int16_t> syy = p2 - p0;
                                    syy.store(buf_ptr_1);
                                }
                            }
                        }

                        if (c < end_c)
                            return cur_node.simple_update(start_r, end_r, c, end_c);

                        return c;
                    };
                }

                cur_n++; // 4
                nodes.push_back(FilterNode());
                nodes[cur_n].op_name = "mag phase quant 1x1";
                nodes[cur_n].op_r = 1;
                nodes[cur_n].op_c = 1;
                nodes[cur_n].parent = cur_n - 1;
                nodes[cur_n].op_type = CV_8U;
                nodes[cur_n].simd_step = mipp::N<int8_t>();
                nodes[cur_n].use_simd = false;
                {
                    nodes[cur_n].simple_update = [&nodes, cur_n, mag_thresh_l2](int start_r, int end_r, int start_c, int end_c) {
                        auto &cur_node = nodes[cur_n];
                        auto &parent_node = nodes[cur_node.parent];

                        const int32_t TG1125 = std::round(std::tan(11.25/180*CV_PI)*(1<<15));
                        const int32_t TG3375 = std::round(std::tan(33.75/180*CV_PI)*(1<<15));
                        const int32_t TG5625 = std::round(std::tan(56.25/180*CV_PI)*(1<<15));
                        const int32_t TG7875 = std::round(std::tan(78.75/180*CV_PI)*(1<<15));

                        int c = start_c;
                        for (int r = start_r; r < end_r; r++){
                            c = start_c;
                            uint8_t *buf_ptr = cur_node.ptr<uint8_t>(r, c);
                            int16_t *parent_buf_ptr_0 = parent_node.ptr<int16_t>(r, c, 0);
                            int16_t *parent_buf_ptr_1 = parent_node.ptr<int16_t>(r, c, 1);
                            for (; c < end_c; c++, buf_ptr++, parent_buf_ptr_0++, parent_buf_ptr_1++){
                                int32_t dx = int32_t(*parent_buf_ptr_0);
                                int32_t dy = int32_t(*parent_buf_ptr_1);
                                int32_t mag = dx*dx+dy*dy;
                                if(mag > mag_thresh_l2){
                                    bool is_0_90 = (dx > 0 && dy > 0) || (dx < 0 && dy < 0);
                                    int32_t x = (dx < 0) ? -dx : dx;
                                    int32_t y = ((dy < 0) ? -dy : dy) << 15;
                                    uint8_t label =  (y<x*TG3375)?
                                                    ((y<x*TG1125)?(0):(1)):
                                                    ((y<x*TG5625)?(2):
                                                    ((y<x*TG7875)?(3):(4)));

                                    label = (label==0 || is_0_90) ? label: 8-label;

                                    // test
    //                                double theta = std::atan2(dy, dx);
    //                                double theta_deg = theta / CV_PI * 180;
    //                                while(theta_deg < 0) theta_deg += 360;
    //                                while(theta_deg > 360) theta_deg -= 360;
    //                                uint8_t angle = std::round(theta_deg * 16.0 / 360.0);
    //                                uint8_t right_label = angle & 7;
    //                                assert(label == right_label);

                                    *buf_ptr = use_hist3x3 ? label: uint8_t(uint8_t(1)<<label);
                                }
                            }
                        }
                        return c;
                    };
                    nodes[cur_n].simd_update = [&nodes, cur_n, imgCols, mag_thresh_l2](int start_r, int end_r, int start_c, int end_c) {
                        auto &cur_node = nodes[cur_n];
                        auto &parent_node = nodes[cur_node.parent];

                        const int32_t TG1125 = std::round(std::tan(11.25/180*CV_PI)*(1<<15));
                        const int32_t TG3375 = std::round(std::tan(33.75/180*CV_PI)*(1<<15));
                        const int32_t TG5625 = std::round(std::tan(56.25/180*CV_PI)*(1<<15));
                        const int32_t TG7875 = std::round(std::tan(78.75/180*CV_PI)*(1<<15));

                        mipp::Reg<int16_t> ZERO16_v = int16_t(0);
                        mipp::Reg<int32_t> ZERO32_v = int32_t(0);
                        mipp::Reg<int32_t> ONE32_v = int32_t(1);
                        mipp::Reg<int32_t> TWO32_v = int32_t(2);
                        mipp::Reg<int32_t> THREE32_v = int32_t(3);
                        mipp::Reg<int32_t> FOUR32_v = int32_t(4);
                        mipp::Reg<int32_t> EIGHT32_v = int32_t(8);
                        mipp::Reg<int32_t> TG1125_v = TG1125;
                        mipp::Reg<int32_t> TG3375_v = TG3375;
                        mipp::Reg<int32_t> TG5625_v = TG5625;
                        mipp::Reg<int32_t> TG7875_v = TG7875;

                        int c = start_c;
                        for (int r = start_r; r < end_r; r++){
                            c = start_c;
                            uint8_t *buf_ptr = cur_node.ptr<uint8_t>(r, c);
                            int16_t *parent_buf_ptr_0 = parent_node.ptr<int16_t>(r, c, 0);
                            int16_t *parent_buf_ptr_1 = parent_node.ptr<int16_t>(r, c, 1);
                            for (; c < end_c; c += cur_node.simd_step, buf_ptr += cur_node.simd_step,
                                 parent_buf_ptr_0 += cur_node.simd_step, parent_buf_ptr_1 += cur_node.simd_step){
                                if (c + 2*cur_node.simd_step >= end_c)
                                    break; // simd may excel end_c, but avoid simd out of img

                                // int16_t step is 2*int32_t_step
                                for(int i=0; i<cur_node.simd_step; i+=cur_node.simd_step/2){
                                    mipp::Reg<int16_t> dx_int16(parent_buf_ptr_0 + i);
                                    mipp::Reg<int16_t> dy_int16(parent_buf_ptr_1 + i);

                                    mipp::Reg<int32_t> dx_int32 = mipp::cvt<int16_t, int32_t>(mipp::low<int16_t>(dx_int16.r));
                                    mipp::Reg<int32_t> dy_int32 = mipp::cvt<int16_t, int32_t>(mipp::low<int16_t>(dy_int16.r));

                                    mipp::Reg<int32_t> mag = dx_int32*dx_int32 + dy_int32*dy_int32;

                                    auto mag_mask = mag > mag_thresh_l2;
                                    if(!mipp::testz(mag_mask)){
                                        mipp::Reg<int32_t> abs_dx = mipp::abs(dx_int32);
                                        mipp::Reg<int32_t> abs_dy = mipp::abs(dy_int32) << 15;

            //                            uint8_t label =  (y<x*TG3375)?
            //                                            ((y<x*TG1125)?(0):(1)):
            //                                            ((y<x*TG5625)?(2):
            //                                            ((y<x*TG7875)?(3):(4)));
                                        mipp::Reg<int32_t> label_v =
                                                    mipp::blend(
                                                    mipp::blend(ZERO32_v,
                                                                ONE32_v    , abs_dy<abs_dx*TG1125_v),
                                                    mipp::blend(TWO32_v,
                                                    mipp::blend(THREE32_v,
                                                                FOUR32_v
                                                                           , abs_dy<abs_dx*TG7875_v)
                                                                           , abs_dy<abs_dx*TG5625_v)
                                                                           , abs_dy<abs_dx*TG3375_v);
                                        label_v = mipp::blend(label_v, EIGHT32_v-label_v,
                                                              ((label_v == ZERO32_v) |
                                                               ((dx_int32>0 & dy_int32>0)|(dx_int32<0 & dy_int32<0))));

                                        label_v = mipp::blend(label_v, ZERO32_v, mag_mask);

                                        // range 0-7, so no worry for signed int while pack
                                        // two lanes may save one pack?
                                        mipp::Reg<int16_t> label16_v = mipp::pack<int32_t, int16_t>(label_v, ZERO32_v);
                                        mipp::Reg<int8_t> label8_v = mipp::pack<int16_t, int8_t>(label16_v, ZERO16_v);

                                        if(use_hist3x3){
                                            label8_v.store((int8_t*)buf_ptr+i);
                                        }else{
                                            int8_t temp_result[mipp::N<int8_t>()] = {0};
                                            label8_v.store(temp_result);
                                            for(int j=0; j<cur_node.simd_step/2; j++){
                                                uint8_t cur_offset = uint8_t(temp_result[j]);
                                                *(buf_ptr+i+j) = uint8_t(uint8_t(1)<<cur_offset);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        if (c < end_c)
                            return cur_node.simple_update(start_r, end_r, c, end_c);
                        return c;
                    };
                }

                if(use_hist3x3){
                    cur_n++;
                    nodes.resize(nodes.size() + 1);
                    nodes[cur_n].op_name = "hist 3x3";
                    nodes[cur_n].op_r = 3;
                    nodes[cur_n].op_c = 3;
                    nodes[cur_n].parent = cur_n - 1;
                    nodes[cur_n].op_type = CV_8U;
                    nodes[cur_n].simd_step = mipp::N<int8_t>();
                    nodes[cur_n].use_simd = false;
                    {
                        nodes[cur_n].simple_update = [&nodes, cur_n](int start_r, int end_r, int start_c, int end_c) {
                            auto &cur_node = nodes[cur_n];
                            auto &parent_node = nodes[cur_node.parent];
                            int c = start_c;
                            const int col_step = cur_node.buffer_cols;
                            for (int r = start_r; r < end_r; r++){
                                c = start_c;
                                uint8_t *buf_ptr = cur_node.ptr<uint8_t>(r, c);
                                uint8_t *parent_buf_ptr = parent_node.ptr<uint8_t>(r, c);
                                for (; c < end_c; c++, buf_ptr++, parent_buf_ptr++){
                                    uint8_t votes_of_ori[8] = {0};
                                    votes_of_ori[*parent_buf_ptr]++;
                                    votes_of_ori[*(parent_buf_ptr+1)]++;
                                    votes_of_ori[*(parent_buf_ptr-1)]++;
                                    votes_of_ori[*(parent_buf_ptr-col_step)]++;
                                    votes_of_ori[*(parent_buf_ptr-col_step+1)]++;
                                    votes_of_ori[*(parent_buf_ptr-col_step-1)]++;
                                    votes_of_ori[*(parent_buf_ptr+col_step)]++;
                                    votes_of_ori[*(parent_buf_ptr+col_step+1)]++;
                                    votes_of_ori[*(parent_buf_ptr+col_step-1)]++;

                                    // Find bin with the most votes from the patch
                                    int max_votes = 0;
                                    int index = -1;
                                    for (int i = 0; i < 8; ++i){
                                        if (max_votes < votes_of_ori[i]){
                                            index = i;
                                            max_votes = votes_of_ori[i];
                                        }
                                    }

                                    // Only accept the quantization if majority of pixels in the patch agree
                                    static const int NEIGHBOR_THRESHOLD = 5;
                                    if (max_votes >= NEIGHBOR_THRESHOLD) *buf_ptr = uint8_t(uint8_t(1) << index);
                                }
                            }
                            return c;
                        };

                        nodes[cur_n].simd_update = nodes[cur_n].simple_update;
                        // nodes[cur_n].simd_update = [imgCols, cur_node](int start_r, int end_r, int start_c, int end_c) {
                        //     auto &parent_node = *cur_node.parent;
                        //     int c = start_c;
                        //     for (int r = start_r; r < end_r; r++)
                        //     {
                        //         c = start_c;
                        //         uint8_t *buf_ptr = cur_node.ptr<uint8_t>(r, c);
                        //         uint8_t *parent_buf_ptr = parent_node.ptr<uint8_t>(r, c);
                        //         for (; c < end_c; c += cur_node.simd_step, buf_ptr += cur_node.simd_step, parent_buf_ptr += cur_node.simd_step)
                        //         {
                        //             if (c + cur_node.simd_step >= end_c)
                        //                 break; // simd may excel end_c, but avoid simd out of img
                        //         }
                        //     }

                        //     if (c < end_c)
                        //         return cur_node.simple_update(start_r, end_r, c, end_c);
                        //     return c;
                        // };
                    }
                }

                cur_n++; // 5
                nodes.push_back(FilterNode());
                nodes[cur_n].op_name = "spread 1x5";
                nodes[cur_n].op_r = 1;
                nodes[cur_n].op_c = cur_T + 1;
                nodes[cur_n].parent = cur_n - 1;
                nodes[cur_n].op_type = CV_8U;
                nodes[cur_n].simd_step = mipp::N<int8_t>();
                nodes[cur_n].use_simd = true;
                {
                    nodes[cur_n].simple_update = [&nodes, cur_n](int start_r, int end_r, int start_c, int end_c) {
                        auto &cur_node = nodes[cur_n];
                        auto &parent_node = nodes[cur_node.parent];
                        int c = start_c;
                        for (int r = start_r; r < end_r; r++){
                            c = start_c;
                            uint8_t *buf_ptr = cur_node.ptr<uint8_t>(r, c);
                            uint8_t *parent_buf_ptr = parent_node.ptr<uint8_t>(r, c);
                            for (; c < end_c; c++, buf_ptr++, parent_buf_ptr++){
                                uint8_t local_sum = *(parent_buf_ptr);
                                for(int i=1; i<=cur_node.op_c/2; i++){
                                    local_sum |= *(parent_buf_ptr+i);
                                    local_sum |= *(parent_buf_ptr-i);
                                }
                                *buf_ptr = local_sum;
                            }
                        }
                        return c;
                    };
                    nodes[cur_n].simd_update = [&nodes, cur_n, imgCols](int start_r, int end_r, int start_c, int end_c) {
                        auto &cur_node = nodes[cur_n];
                        auto &parent_node = nodes[cur_node.parent];
                        int c = start_c;
                        for (int r = start_r; r < end_r; r++){
                            c = start_c;
                            uint8_t *buf_ptr = cur_node.ptr<uint8_t>(r, c);
                            uint8_t *parent_buf_ptr = parent_node.ptr<uint8_t>(r, c);
                            for (; c < end_c; c += cur_node.simd_step, buf_ptr += cur_node.simd_step,
                                 parent_buf_ptr += cur_node.simd_step){
                                if (c + cur_node.simd_step >= end_c)
                                    break; // simd may excel end_c, but avoid simd out of img

                                mipp::Reg<uint8_t> local_sum(parent_buf_ptr);
                                for(int i=1; i<=cur_node.op_c/2; i++){
                                    mipp::Reg<uint8_t> src_v0(parent_buf_ptr+i);
                                    local_sum = mipp::orb(src_v0, local_sum);

                                    mipp::Reg<uint8_t> src_v1(parent_buf_ptr-i);
                                    local_sum = mipp::orb(src_v1, local_sum);
                                }
                                local_sum.store(buf_ptr);
                            }
                        }

                        if (c < end_c)
                            return cur_node.simple_update(start_r, end_r, c, end_c);
                        return c;
                    };
                }

                cur_n++; // 6
                nodes.push_back(FilterNode());
                nodes[cur_n].op_name = "spread 5x1";
                nodes[cur_n].op_r = cur_T + 1;
                nodes[cur_n].op_c = 1;
                nodes[cur_n].parent = cur_n - 1;
                nodes[cur_n].op_type = CV_8U;
                nodes[cur_n].simd_step = mipp::N<int8_t>();
                nodes[cur_n].use_simd = false;
                {
                    nodes[cur_n].simple_update = [&nodes, cur_n](int start_r, int end_r, int start_c, int end_c) {
                        auto &cur_node = nodes[cur_n];
                        auto &parent_node = nodes[cur_node.parent];

                        int c = start_c;
                        for (int r = start_r; r < end_r; r++){
                            c = start_c;
                            uint8_t *buf_ptr = cur_node.ptr<uint8_t>(r, c);

                            uint8_t *parent_buf_ptr[cur_node.op_r];
                            uint8_t** parent_buf_ptr_center = parent_buf_ptr + cur_node.op_r/2;
                            parent_buf_ptr_center[0] = parent_node.ptr<uint8_t>(r, c);
                            for(int i=1; i<=cur_node.op_r/2; i++){
                                parent_buf_ptr_center[+i] = parent_node.ptr<uint8_t>(r+i, c);
                                parent_buf_ptr_center[-i] = parent_node.ptr<uint8_t>(r-i, c);
                            }

                            for (; c < end_c; c++, buf_ptr++){
                                uint8_t local_sum =  *(parent_buf_ptr_center[0]);
                                for(int i=1; i<=cur_node.op_r/2; i++){
                                    local_sum |= *(parent_buf_ptr_center[i]);
                                    local_sum |= *(parent_buf_ptr_center[-i]);
                                }
                                *buf_ptr = local_sum;

                                for(int i=0; i<cur_node.op_r; i++)
                                    parent_buf_ptr[i]++;
                            }
                        }
                        return c;
                    };
                    nodes[cur_n].simd_update = [&nodes, cur_n, imgCols](int start_r, int end_r, int start_c, int end_c) {
                        auto &cur_node = nodes[cur_n];
                        auto &parent_node = nodes[cur_node.parent];

                        int c = start_c;
                        for (int r = start_r; r < end_r; r++){
                            c = start_c;
                            uint8_t *buf_ptr = cur_node.ptr<uint8_t>(r, c);

                            uint8_t *parent_buf_ptr[cur_node.op_r];
                            uint8_t** parent_buf_ptr_center = parent_buf_ptr + cur_node.op_r/2;
                            parent_buf_ptr_center[0] = parent_node.ptr<uint8_t>(r, c);
                            for(int i=1; i<=cur_node.op_r/2; i++){
                                parent_buf_ptr_center[+i] = parent_node.ptr<uint8_t>(r+i, c);
                                parent_buf_ptr_center[-i] = parent_node.ptr<uint8_t>(r-i, c);
                            }

                            for (; c < end_c; c += cur_node.simd_step, buf_ptr += cur_node.simd_step){
                                if (c + cur_node.simd_step >= end_c)
                                    break; // simd may excel end_c, but avoid simd out of img

                                mipp::Reg<uint8_t> local_sum(parent_buf_ptr_center[0]);
                                for(int i=1; i<=cur_node.op_c/2; i++){
                                    mipp::Reg<uint8_t> src_v0(parent_buf_ptr_center[i]);
                                    local_sum = mipp::orb(src_v0, local_sum);

                                    mipp::Reg<uint8_t> src_v1(parent_buf_ptr_center[-i]);
                                    local_sum = mipp::orb(src_v1, local_sum);
                                }
                                local_sum.store(buf_ptr);
                                for(int i=0; i<cur_node.op_r; i++)
                                    parent_buf_ptr[i]+=cur_node.simd_step;
                            }
                        }

                        if (c < end_c)
                            return cur_node.simple_update(start_r, end_r, c, end_c);
                        return c;
                    };
                }

                cur_n++; // 7
                nodes.push_back(FilterNode());
                nodes[cur_n].op_name = "response 1x1";
                nodes[cur_n].op_r = 1;
                nodes[cur_n].op_c = 1;
                nodes[cur_n].parent = cur_n - 1;
                nodes[cur_n].num_buf = 8;
                nodes[cur_n].op_type = CV_8U;
                nodes[cur_n].simd_step = mipp::N<int8_t>();
                nodes[cur_n].use_simd = false;
                {
                    const uint8_t scores[2] = {4, 1};
                    const uint8_t hit_mask[8] = { 1,   2, 4,  8,  16, 32, 64,  128};
                    const uint8_t side_mask[8] = {130, 5, 10, 20, 40, 80, 160, 65};
                    nodes[cur_n].simple_update = [&nodes, cur_n, hit_mask, side_mask, scores](int start_r, int end_r, int start_c, int end_c) {
                        auto &cur_node = nodes[cur_n];
                        auto &parent_node = nodes[cur_node.parent];
                        int c = start_c;
                        for (int ori = 0; ori < 8; ori++){
                            for (int r = start_r; r < end_r; r++){
                                c = start_c;
                                uint8_t *buf_ptr = cur_node.ptr<uint8_t>(r, c, ori);
                                uint8_t *parent_buf_ptr = parent_node.ptr<uint8_t>(r, c);
                                for (; c < end_c; c++, buf_ptr++, parent_buf_ptr++){
                                    *buf_ptr = (hit_mask[ori] & *parent_buf_ptr) ? scores[0] :
                                    ((side_mask[ori] & *parent_buf_ptr) ? scores[1] : 0);
                                }
                            }
                        }
                        return c;
                    };
                    nodes[cur_n].simd_update = [&nodes, cur_n, imgCols, hit_mask, side_mask, scores](int start_r, int end_r, int start_c, int end_c) {
                        auto &cur_node = nodes[cur_n];
                        auto &parent_node = nodes[cur_node.parent];

                        mipp::Reg<uint8_t> const_score0 = scores[0];
                        mipp::Reg<uint8_t> const_score1 = scores[1];
                        mipp::Reg<uint8_t> zero8_v = uint8_t(0);
                        int c = start_c;
                        for (int ori = 0; ori < 8; ori++){
                            mipp::Reg<uint8_t> hit_mask_v = hit_mask[ori];
                            mipp::Reg<uint8_t> side_mask_v = side_mask[ori];
                            for (int r = start_r; r < end_r; r++){
                                c = start_c;
                                uint8_t *buf_ptr = cur_node.ptr<uint8_t>(r, c, ori);
                                uint8_t *parent_buf_ptr = parent_node.ptr<uint8_t>(r, c);
                                for (; c < end_c; c += cur_node.simd_step, buf_ptr += cur_node.simd_step,
                                     parent_buf_ptr += cur_node.simd_step){
                                    if (c + cur_node.simd_step >= end_c)
                                        break; // simd may excel end_c, but avoid simd out of img

                                    mipp::Reg<uint8_t> src_v(parent_buf_ptr);
                                    auto result = mipp::blend(mipp::blend(zero8_v, const_score1, (side_mask_v & src_v) == zero8_v),
                                        const_score0, (hit_mask_v & src_v) == zero8_v);
                                    result.store(buf_ptr);
                                }
                            }

                            if (c < end_c)
                                return cur_node.simple_update(start_r, end_r, c, end_c);
                        }
                        return c;
                    };
                }

                cur_n++; // 8
                nodes.push_back(FilterNode());
                nodes[cur_n].op_name = "linearize 1x1";
                nodes[cur_n].op_r = 1;
                nodes[cur_n].op_c = 1;
                nodes[cur_n].parent = cur_n - 1;
                nodes[cur_n].num_buf = 0; // last buffer is output, no need to alloc
                nodes[cur_n].op_type = CV_8U;
                nodes[cur_n].simd_step = mipp::N<int8_t>();
                nodes[cur_n].buffers = lm_pyramid[cur_l][0]; // vector<Mat> pass by value, but Mat is by ref
                nodes[cur_n].use_simd = false;
                {
                    const int linearize_row_step = imgCols / cur_T;
                    nodes[cur_n].simple_update = [&nodes, cur_n, linearize_row_step, cur_T](int start_r, int end_r, int start_c, int end_c) {
                        auto &cur_node = nodes[cur_n];
                        auto &parent_node = nodes[cur_node.parent];
                        for(int ori = 0; ori < 8; ori++){
                            // assume cur_T = 4, row col of the linearized response_map:
                            // int lrs = linearize_row_step;
                            //       0            1            2            3             lrs - 1
                            //  0  1  2  3   0  1  2  3   0  1  2  3   0  1  2  3  ...
                            //  4  5  6  7   4  5  6  7   4  5  6  7   4  5  6  7  ...
                            //  8  9 10 11   8  9 10 11   8  9 10 11   8  9 10 11  ...
                            // 12 13 14 15  12 13 14 15  12 13 14 15  12 13 14 15  ...
                            // ----------------------------------------------------
                            //     lrs+0        lrs+1        lrs+2       lrs+3            2*lrs - 1
                            //  0  1  2  3   0  1  2  3   0  1  2  3   0  1  2  3  ...
                            //  4  5  6  7   4  5  6  7   4  5  6  7   4  5  6  7  ...
                            //  8  9 10 11   8  9 10 11   8  9 10 11   8  9 10 11  ...
                            // 12 13 14 15  12 13 14 15  12 13 14 15  12 13 14 15  ...
                            // ----------------------------------------------------
                            //   2*lrs+0      2*lrs+1       2*lrs+2     2*lrs+3           3*lrs - 1
                            //  0  1  2  3   0  1  2  3   0  1  2  3   0  1  2  3  ...
                            //  4  5  6  7   4  5  6  7   4  5  6  7   4  5  6  7  ...
                            //  8  9 10 11   8  9 10 11   8  9 10 11   8  9 10 11  ...
                            // 12 13 14 15  12 13 14 15  12 13 14 15  12 13 14 15  ...

                            // so :
    //                         int target_c = linearize_row_step * (r / cur_T) + c / cur_T;
    //                         int target_r = cur_T * (r % cur_T) + c % cur_T;

                             // cleaner codes, but heavy math operation in the innermost loop
    //                         int c = start_c;
    //                         for (int r = start_r; r < end_r; r++){
    //                             c = start_c;
    //                             uint8_t *parent_buf_ptr = parent_node.ptr<uint8_t>(r, c, ori);
    //                             for (; c < end_c; c++, parent_buf_ptr++){
    //                                 int target_c = linearize_row_step * (r / cur_T) + c / cur_T;
    //                                 int target_r = cur_T * (r % cur_T) + c % cur_T;
    //                                 cur_node.buffers[ori].at<uint8_t>(target_r, target_c) = *parent_buf_ptr;
    //                             }
    //                         }

                            // more codes, but less operation int the innermost loop
                            assert(start_c % cur_T == 0);
                            int target_start_c = linearize_row_step * (start_r / cur_T) + start_c / cur_T;
                            int target_start_r = cur_T * (start_r % cur_T) + start_c % cur_T;

                            for(int tileT_r = start_r; tileT_r < start_r + cur_T; ++tileT_r){
                                for (int tileT_c = start_c; tileT_c < start_c + cur_T; ++tileT_c){
                                    uint8_t *memory = &cur_node.buffers[ori].at<uint8_t>(target_start_r, target_start_c);
                                    target_start_r++;
                                    if(target_start_r >= cur_T * cur_T){
                                        target_start_r -= cur_T * cur_T;
                                        target_start_c += linearize_row_step;
                                    }

                                    // Inner two loops copy every T-th pixel into the linear memory
                                    for(int r = tileT_r; r < end_r; r += cur_T, memory += linearize_row_step){
                                        uint8_t *parent_buf_ptr = parent_node.buffers[ori].ptr(r%parent_node.buffer_rows);
                                        uint8_t *local_memory = memory;
                                        for (int c = tileT_c; c < end_c; c += cur_T)
                                            *local_memory++ = parent_buf_ptr[c];
                                    }
                                }
                            }
                        }
                        return end_c;
                    };
                }

                nodes[cur_n].backward_rc(nodes, imgRows, imgCols, 0, 0);
//                nodes[cur_n].backward_rc(nodes, tileRows, imgCols, 0, 0); // cycle buffer to save some space

                auto to_upper_pow2 = [](uint32_t x){
                    int power = 1;
                    while(power < x)
                        power*=2;
                    return power;
                };

                for (auto &node : nodes){
                    for (int i = 0; i < node.num_buf; i++){
                        // node.buffer_rows = to_upper_pow2(node.buffer_rows);  // may be better for mod operation?
                        node.buffers.push_back(cv::Mat(node.buffer_rows, node.buffer_cols, node.op_type, cv::Scalar(0)));
                    }

                    node.anchor_col = -node.padded_cols;
                    node.anchor_row = tile_start_rows - node.padded_rows;
                }
                // for tile_r tile_c
                for(int tile_r = tile_start_rows; tile_r < tile_end_rows; tile_r += tileRows){
                    int end_r = tile_r + tileRows;
                    end_r = (end_r > tile_end_rows) ? tile_end_rows : end_r;

                    for(int tile_c = 0; tile_c < imgCols; tile_c += tileCols){
                        int end_c = tile_c + tileCols;
                        end_c = (end_c > imgCols) ? imgCols : end_c;

                        for(int cur_node = 0; cur_node < nodes.size(); cur_node++){

                            // clamp row col
                            int update_start_r = tile_r - nodes[cur_node].padded_rows;
                            if (update_start_r < nodes[cur_node].prepared_row)
                                update_start_r = nodes[cur_node].prepared_row;
                            int update_end_r = end_r + nodes[cur_node].padded_rows;
                            if (update_end_r > imgRows)
                                update_end_r = imgRows;

                            int update_start_c = tile_c - nodes[cur_node].padded_cols;
                            if (update_start_c < nodes[cur_node].prepared_col)
                                update_start_c = nodes[cur_node].prepared_col;
                            int update_end_c = end_c + nodes[cur_node].padded_cols;
                            if (update_end_c > imgCols)
                                update_end_c = imgCols;

                            if (nodes[cur_node].use_simd){
                                nodes[cur_node].prepared_col =
                                    nodes[cur_node].simd_update(update_start_r, update_end_r, update_start_c, update_end_c);
                            }else{
                                nodes[cur_node].prepared_col =
                                    nodes[cur_node].simple_update(update_start_r, update_end_r, update_start_c, update_end_c);
                            }
                        }
                    }

                    for (int cur_node = 0; cur_node < nodes.size(); cur_node++){
                        nodes[cur_node].prepared_col = 0;
                        nodes[cur_node].prepared_row = end_r + nodes[cur_node].padded_rows;
                    }

                    // test for update order
    //                Mat pyr_show;
    //                convertScaleAbs(nodes[3].buffers[0], pyr_show, 0.25);
    //                cvtColor(pyr_show, pyr_show, CV_GRAY2BGR);
    //                int padding = 100;
    //                Mat padded_show(pyr_show.rows+padding*2, pyr_show.cols+padding*2, CV_8UC3, cv::Scalar(0, 255, 0));
    //                pyr_show.copyTo(padded_show({padding, padding, pyr_show.cols, pyr_show.rows}));
    //                imshow("pyr_src", padded_show);
    //                waitKey(100);
                }

                const bool debug_ = false;
                if(debug_){
                    // gaussian test
                    Mat gauss_src;
                    GaussianBlur(source, gauss_src, {5, 5}, 0);
                    gauss_src.convertTo(gauss_src, CV_32F);
                    Mat fusion_gauss_src = nodes[1].buffers[0](
                                Rect(nodes[1].padded_cols, nodes[1].padded_rows, gauss_src.cols, gauss_src.rows)).clone();
                    fusion_gauss_src.convertTo(fusion_gauss_src, CV_32F);
                    Mat diff_gauss = cv::abs(gauss_src - fusion_gauss_src);
                    imshow("diff gauss", diff_gauss > 1); // >1 avoid float int round issue
    //                waitKey(0);

                    // sobel x/y test
                    Mat dx, dy;
                    cv::Sobel(fusion_gauss_src, dx, CV_32F, 1, 0);
                    cv::Sobel(fusion_gauss_src, dy, CV_32F, 0, 1);

                    Mat fusion_dx = nodes[3].buffers[0](
                                Rect(nodes[3].padded_cols, nodes[3].padded_rows, dx.cols, dx.rows)).clone();
                    fusion_dx.convertTo(fusion_dx, CV_32F);
                    Mat fusion_dy = nodes[3].buffers[1](
                                Rect(nodes[3].padded_cols, nodes[3].padded_rows, dy.cols, dy.rows)).clone();
                    fusion_dy.convertTo(fusion_dy, CV_32F);

                    Mat diff_dx = cv::abs(dx - fusion_dx);
                    Mat diff_dy = cv::abs(dy - fusion_dy);

                    imshow("diff dx", diff_dx > 1);
                    imshow("diff dy", diff_dy > 1);
    //                waitKey(0);

                    // mag quant test
                    Mat angle;
                    phase(fusion_dx, fusion_dy, angle, true);
                    Mat magnitude = fusion_dx.mul(fusion_dx) + fusion_dy.mul(fusion_dy);
                    Mat quantized_unfiltered;
                    angle.convertTo(quantized_unfiltered, CV_8U, 16.0 / 360.0);
                    // Mask 16 buckets into 8 quantized orientations
                    for (int r = 1; r < angle.rows - 1; ++r){
                        uchar *quant_r = quantized_unfiltered.ptr<uchar>(r);
                        for (int c = 1; c < angle.cols - 1; ++c){
                            quant_r[c] &= 7;
                            quant_r[c] = uint8_t(uint8_t(1) << quant_r[c]);
                            if(magnitude.at<float>(r, c) <= mag_thresh_l2){
                                quant_r[c] = 0;
                            }
                        }
                    }
                    Mat fusion_quant = nodes[4].buffers[0](
                                Rect(nodes[4].padded_cols, nodes[4].padded_rows, dy.cols, dy.rows)).clone();
                    Mat diff_quant = fusion_quant != quantized_unfiltered;
                    imshow("diff quant", diff_quant); // small diff due to mag float or int32 compare


                    // spread test, no offset
                    Mat spread_quant(fusion_quant.size(), CV_8U, Scalar(0));
                    for(int r=cur_T/2; r<spread_quant.rows-cur_T/2; r++){
                        for(int c=cur_T/2; c<spread_quant.cols-cur_T/2; c++){
                            uint8_t& res = spread_quant.at<uint8_t>(r, c);
                            for(int i=-cur_T/2; i<=cur_T/2; i++){
                                for(int j=-cur_T/2; j<=cur_T/2; j++){
                                    res |= fusion_quant.at<uint8_t>(r+i, c+j);
                                }
                            }
                        }
                    }
                    Mat fusion_spread_quant = nodes[6].buffers[0](
                                Rect(nodes[6].padded_cols, nodes[6].padded_rows, dy.cols, dy.rows)).clone();
                    Mat diff_spread = fusion_spread_quant != spread_quant;
                    imshow("diff spread", diff_spread);
                    waitKey(0);

                    // response test
                    assert(nodes[7].buffers[0].rows = imgRows);
                    assert(nodes[7].buffers[0].cols = imgCols);

                    auto leftRot = [](uint8_t n, unsigned d){ //rotate n by d bits
                         return uint8_t(n << d)|(n >> (8 - d));
                    };
                    auto rightRot = [](uint8_t n, unsigned d){ //rotate n by d bits
                         return uint8_t(n >> d)|(n << (8 - d));
                    };

                    for (int ori = 0; ori < 8; ori++){
                        Mat response(imgRows, imgCols, CV_8U, cv::Scalar(0));

                        uint8_t cur_ori = uint8_t(uint8_t(1) << ori);
                        uint8_t side_ori = leftRot(cur_ori, 1) | rightRot(cur_ori, 1);

                        for(int r=0; r<response.rows; r++){
                            for(int c=0; c<response.cols; c++){
                                uint8_t& res = response.at<uint8_t>(r, c);
                                uint8_t cur_quant_ori = fusion_spread_quant.at<uint8_t>(r, c);

                                if(cur_quant_ori & cur_ori) res = 4;
                                else if(side_ori & cur_quant_ori) res = 1;
                                else res = 0;
                            }
                        }
                        Mat diff_res =  response != nodes[7].buffers[ori];
                        assert(cv::sum(diff_res)[0] == 0);
                        imshow("diff res" + to_string(ori), diff_res);
                    }

                    //linearize test
                    for (int ori = 0; ori < 8; ori++){
                        Mat before_linear = nodes[7].buffers[ori];
                        Mat linearized;
                        linearize(before_linear, linearized, cur_T);
                        Mat diff_linear = linearized != nodes[8].buffers[ori]; // too long to show, just test zero
                        assert(cv::sum(diff_linear)[0] == 0);
                    }

                    waitKey(0);

                    exit(1);
                }

            }
        }
    }else{  // original method
        // Initialize each ColorGradient with our sources
        std::vector<Ptr<ColorGradientPyramid>> quantizers;
        CV_Assert(mask.empty() || mask.size() == source.size());
        quantizers.push_back(modality->process(source, mask));

        for (int l = 0; l < pyramid_levels; ++l){
            int T = T_at_level[l];
            std::vector<LinearMemories> &lm_level = lm_pyramid[l];

            if (l > 0){
                for (int i = 0; i < (int)quantizers.size(); ++i)
                    quantizers[i]->pyrDown();
            }

            Mat quantized, spread_quantized;
            std::vector<Mat> response_maps;
            for (int i = 0; i < (int)quantizers.size(); ++i){
                quantizers[i]->quantize(quantized);
                spread(quantized, spread_quantized, T);
                computeResponseMaps(spread_quantized, response_maps);

                LinearMemories &memories = lm_level[i];
                for (int j = 0; j < 8; ++j)
                    linearize(response_maps[j], memories[j], T);
            }
            sizes.push_back(quantized.size());
        }
    }
    // ----------------------------------------------------------------------


    if (class_ids.empty()){
        // Match all templates
        TemplatesMap::const_iterator it = class_templates.begin(), itend = class_templates.end();
        for (; it != itend; ++it)
            matchClass(lm_pyramid, sizes, threshold, matches, it->first, it->second);
    }
    else{
        // Match only templates for the requested class IDs
        for (int i = 0; i < (int)class_ids.size(); ++i){
            TemplatesMap::const_iterator it = class_templates.find(class_ids[i]);
            if (it != class_templates.end())
                matchClass(lm_pyramid, sizes, threshold, matches, it->first, it->second);
        }
    }

    // Sort matches by similarity, and prune any duplicates introduced by pyramid refinement
    std::sort(matches.begin(), matches.end());
    std::vector<Match>::iterator new_end = std::unique(matches.begin(), matches.end());
    matches.erase(new_end, matches.end());

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
#pragma omp declare reduction \
    (omp_insert: std::vector<Match>: omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))

#pragma omp parallel for reduction(omp_insert:matches)
    for (size_t template_id = 0; template_id < template_pyramids.size(); ++template_id)
    {
        const TemplatePyramid &tp = template_pyramids[template_id];
        // First match over the whole image at the lowest pyramid level
        /// @todo Factor this out into separate function
        const std::vector<LinearMemories> &lowest_lm = lm_pyramid.back();

        std::vector<Match> candidates;
        {
            // Compute similarity maps for each ColorGradient at lowest pyramid level
            Mat similarities;
            int lowest_start = static_cast<int>(tp.size() - 1);
            int lowest_T = T_at_level.back();
            int num_features = 0;

            {
                const Template &templ = tp[lowest_start];
                num_features += static_cast<int>(templ.features.size());

                if (templ.features.size() < 64){
                    similarity_64(lowest_lm[0], templ, similarities, sizes.back(), lowest_T);
                    similarities.convertTo(similarities, CV_16U);
                }else if (templ.features.size() < 8192){
                    similarity(lowest_lm[0], templ, similarities, sizes.back(), lowest_T);
                }else{
                    CV_Error(Error::StsBadArg, "feature size too large");
                }
            }

            // Find initial matches
            for (int r = 0; r < similarities.rows; ++r)
            {
                ushort *row = similarities.ptr<ushort>(r);
                for (int c = 0; c < similarities.cols; ++c)
                {
                    int raw_score = row[c];
                    float score = (raw_score * 100.f) / (4 * num_features);

                    if (score > threshold)
                    {
                        int offset = /*lowest_T / 2 + */(lowest_T % 2 - 1); // spread has no offset now
                        int x = c * lowest_T + offset;
                        int y = r * lowest_T + offset;
                        candidates.push_back(Match(x, y, score, class_id, static_cast<int>(template_id)));
                    }
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
            int offset = /*T / 2 +*/ (T % 2 - 1); // spread has no offset now
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

                {
                    const Template &templ = tp[start];
                    numFeatures += static_cast<int>(templ.features.size());

                    if (templ.features.size() < 64){
                        similarityLocal_64(lms[0], templ, similarities2, size, T, Point(x, y));
                        similarities2.convertTo(similarities2, CV_16U);
                    }else if (templ.features.size() < 8192){
                        similarityLocal(lms[0], templ, similarities2, size, T, Point(x, y));
                    }else{
                        CV_Error(Error::StsBadArg, "feature size too large");
                    }
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
