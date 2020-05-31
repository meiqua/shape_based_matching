#include "line2Dup.h"
#include <memory>
#include <iostream>
#include <assert.h>
#include <chrono>

#include "cuda_icp/icp.h"

using namespace std;
using namespace cv;

static std::string prefix = "/home/meiqua/shape_based_matching/test/";

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
        std::cout << message << "\nelasped time:" << t << "s" << std::endl;
        reset();
    }
private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1> > second_;
    std::chrono::time_point<clock_> beg_;
};
// NMS, got from cv::dnn so we don't need opencv contrib
// just collapse it
namespace  cv_dnn {
namespace
{

template <typename T>
static inline bool SortScorePairDescend(const std::pair<float, T>& pair1,
                          const std::pair<float, T>& pair2)
{
    return pair1.first > pair2.first;
}

} // namespace

inline void GetMaxScoreIndex(const std::vector<float>& scores, const float threshold, const int top_k,
                      std::vector<std::pair<float, int> >& score_index_vec)
{
    for (size_t i = 0; i < scores.size(); ++i)
    {
        if (scores[i] > threshold)
        {
            score_index_vec.push_back(std::make_pair(scores[i], i));
        }
    }
    std::stable_sort(score_index_vec.begin(), score_index_vec.end(),
                     SortScorePairDescend<int>);
    if (top_k > 0 && top_k < (int)score_index_vec.size())
    {
        score_index_vec.resize(top_k);
    }
}

template <typename BoxType>
inline void NMSFast_(const std::vector<BoxType>& bboxes,
      const std::vector<float>& scores, const float score_threshold,
      const float nms_threshold, const float eta, const int top_k,
      std::vector<int>& indices, float (*computeOverlap)(const BoxType&, const BoxType&))
{
    CV_Assert(bboxes.size() == scores.size());
    std::vector<std::pair<float, int> > score_index_vec;
    GetMaxScoreIndex(scores, score_threshold, top_k, score_index_vec);

    // Do nms.
    float adaptive_threshold = nms_threshold;
    indices.clear();
    for (size_t i = 0; i < score_index_vec.size(); ++i) {
        const int idx = score_index_vec[i].second;
        bool keep = true;
        for (int k = 0; k < (int)indices.size() && keep; ++k) {
            const int kept_idx = indices[k];
            float overlap = computeOverlap(bboxes[idx], bboxes[kept_idx]);
            keep = overlap <= adaptive_threshold;
        }
        if (keep)
            indices.push_back(idx);
        if (keep && eta < 1 && adaptive_threshold > 0.5) {
          adaptive_threshold *= eta;
        }
    }
}


// copied from opencv 3.4, not exist in 3.0
template<typename _Tp> static inline
double jaccardDistance__(const Rect_<_Tp>& a, const Rect_<_Tp>& b) {
    _Tp Aa = a.area();
    _Tp Ab = b.area();

    if ((Aa + Ab) <= std::numeric_limits<_Tp>::epsilon()) {
        // jaccard_index = 1 -> distance = 0
        return 0.0;
    }

    double Aab = (a & b).area();
    // distance = 1 - jaccard_index
    return 1.0 - Aab / (Aa + Ab - Aab);
}

template <typename T>
static inline float rectOverlap(const T& a, const T& b)
{
    return 1.f - static_cast<float>(jaccardDistance__(a, b));
}

void NMSBoxes(const std::vector<Rect>& bboxes, const std::vector<float>& scores,
                          const float score_threshold, const float nms_threshold,
                          std::vector<int>& indices, const float eta=1, const int top_k=0)
{
    NMSFast_(bboxes, scores, score_threshold, nms_threshold, eta, top_k, indices, rectOverlap);
}

}

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
        std::cout << message << "\nelasped time:" << t << "s" << std::endl;
        reset();
    }
private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1> > second_;
    std::chrono::time_point<clock_> beg_;
};

void angle_test(string mode = "test", bool viewICP = false){
    line2Dup::Detector detector(128, {4, 8});

    if(mode != "test"){
        Mat img = imread(prefix+"case1/train.png");
        assert(!img.empty() && "check your img path");

        Rect roi(130, 110, 270, 270);
        img = img(roi).clone();
        Mat mask = Mat(img.size(), CV_8UC1, {255});

        // padding to avoid rotating out
        int padding = 100;
        cv::Mat padded_img = cv::Mat(img.rows + 2*padding, img.cols + 2*padding, img.type(), cv::Scalar::all(0));
        img.copyTo(padded_img(Rect(padding, padding, img.cols, img.rows)));

        cv::Mat padded_mask = cv::Mat(mask.rows + 2*padding, mask.cols + 2*padding, mask.type(), cv::Scalar::all(0));
        mask.copyTo(padded_mask(Rect(padding, padding, img.cols, img.rows)));

        shape_based_matching::shapeInfo_producer shapes(padded_img, padded_mask);
        shapes.angle_range = {0, 360};
        shapes.angle_step = 1;

        shapes.scale_range = {1}; // support just one
        shapes.produce_infos();
        std::vector<shape_based_matching::shapeInfo_producer::Info> infos_have_templ;
        string class_id = "test";

        bool is_first = true;

        // for other scales you want to re-extract points: 
        // set shapes.scale_range then produce_infos; set is_first = false;

        int first_id = 0;
        float first_angle = 0;
        for(auto& info: shapes.infos){
            Mat to_show = shapes.src_of(info);

            std::cout << "\ninfo.angle: " << info.angle << std::endl;
            int templ_id;

            if(is_first){
                templ_id = detector.addTemplate(shapes.src_of(info), class_id, shapes.mask_of(info));
                first_id = templ_id;
                first_angle = info.angle;

                if(use_rot) is_first = false;
            }else{
                templ_id = detector.addTemplate_rotate(class_id, first_id,
                                                       info.angle-first_angle,
                                                {shapes.src.cols/2.0f, shapes.src.rows/2.0f});
            }

            auto templ = detector.getTemplates("test", templ_id);
            for(int i=0; i<templ[0].features.size(); i++){
                auto feat = templ[0].features[i];
                cv::circle(to_show, {feat.x+templ[0].tl_x, feat.y+templ[0].tl_y}, 3, {0, 0, 255}, -1);
            }
            
            // will be faster if not showing this
            imshow("train", to_show);
            waitKey(1);

            std::cout << "templ_id: " << templ_id << std::endl;
            if(templ_id != -1){
                infos_have_templ.push_back(info);
            }
        }
        detector.writeClasses(prefix+"case1/%s_templ.yaml");
        shapes.save_infos(infos_have_templ, prefix + "case1/test_info.yaml");
        std::cout << "train end" << std::endl << std::endl;
    }else if(mode=="test"){
        std::vector<std::string> ids;
        ids.push_back("test");
        detector.readClasses(ids, prefix+"case1/%s_templ.yaml");

        // angle & scale are saved here, fetched by match id
        auto infos = shape_based_matching::shapeInfo_producer::load_infos(prefix + "case1/test_info.yaml");

        Mat test_img = imread(prefix+"case1/train.png");
        assert(!test_img.empty() && "check your img path");

        int padding = 100;
        cv::Mat padded_img = cv::Mat(test_img.rows + 2*padding,
                                     test_img.cols + 2*padding, test_img.type(), cv::Scalar::all(0));
        test_img.copyTo(padded_img(Rect(padding, padding, test_img.cols, test_img.rows)));

        int stride = 16;
        int n = padded_img.rows/stride;
        int m = padded_img.cols/stride;
        Rect roi(0, 0, stride*m , stride*n);
        Mat img = padded_img(roi).clone();
        assert(img.isContinuous());

//        cvtColor(img, img, CV_BGR2GRAY);

        std::cout << "test img size: " << img.rows * img.cols << std::endl << std::endl;

        Timer timer;
        auto matches = detector.match(img, 90, ids);
        timer.out();


        std::cout << "matches.size(): " << matches.size() << std::endl;
        size_t top5 = 5;
        if(top5>matches.size()) top5=matches.size();

        // construct scene
        Scene_edge scene;
        // buffer
        vector<::Vec2f> pcd_buffer, normal_buffer;
        scene.init_Scene_edge_cpu(img, pcd_buffer, normal_buffer);

        if(img.channels() == 1) cvtColor(img, img, CV_GRAY2BGR);

        cv::Mat edge_global;  // get edge
        {
            cv::Mat gray;
            if(img.channels() > 1){
                cv::cvtColor(img, gray, CV_BGR2GRAY);
            }else{
                gray = img;
            }

            cv::Mat smoothed = gray;
            cv::Canny(smoothed, edge_global, 30, 60);

            if(edge_global.channels() == 1) cvtColor(edge_global, edge_global, CV_GRAY2BGR);
        }

        for(int i=top5-1; i>=0; i--)
        {
            Mat edge = edge_global.clone();

            auto match = matches[i];
            auto templ = detector.getTemplates("test",
                                               match.template_id);

            // 270 is width of template image
            // 100 is padding when training
            // tl_x/y: template croping topleft corner when training

            float r_scaled = 270/2.0f*infos[match.template_id].scale;

            // scaling won't affect this, because it has been determined by warpAffine
            // cv::warpAffine(src, dst, rot_mat, src.size()); last param
            float train_img_half_width = 270/2.0f + 100;
            float train_img_half_height = 270/2.0f + 100;

            // center x,y of train_img in test img
            float x =  match.x - templ[0].tl_x + train_img_half_width;
            float y =  match.y - templ[0].tl_y + train_img_half_height;

            vector<::Vec2f> model_pcd(templ[0].features.size());
            for(int i=0; i<templ[0].features.size(); i++){
                auto& feat = templ[0].features[i];
                model_pcd[i] = {
                    float(feat.x + match.x),
                    float(feat.y + match.y)
                };
            }
            cuda_icp::RegistrationResult result = cuda_icp::ICP2D_Point2Plane_cpu(model_pcd, scene);

            cv::Vec3b randColor;
            randColor[0] = 0;
            randColor[1] = 0;
            randColor[2] = 255;
            for(int i=0; i<templ[0].features.size(); i++){
                auto feat = templ[0].features[i];
                cv::circle(edge, {feat.x+match.x, feat.y+match.y}, 2, randColor, -1);
            }

            if(viewICP){
                imshow("icp", edge);
                waitKey(0);
            }


            randColor[0] = 0;
            randColor[1] = 255;
            randColor[2] = 0;
            for(int i=0; i<templ[0].features.size(); i++){
                auto feat = templ[0].features[i];
                float x = feat.x + match.x;
                float y = feat.y + match.y;
                float new_x = result.transformation_[0][0]*x + result.transformation_[0][1]*y + result.transformation_[0][2];
                float new_y = result.transformation_[1][0]*x + result.transformation_[1][1]*y + result.transformation_[1][2];

                cv::circle(edge, {int(new_x+0.5f), int(new_y+0.5f)}, 2, randColor, -1);
            }
            if(viewICP){
                imshow("icp", edge);
                waitKey(0);
            }

            double init_angle = infos[match.template_id].angle;
            init_angle = init_angle >= 180 ? (init_angle-360) : init_angle;

            double ori_diff_angle = std::abs(init_angle);
            double icp_diff_angle = std::abs(-std::asin(result.transformation_[1][0])/CV_PI*180 +
                    init_angle);
            double improved_angle = ori_diff_angle - icp_diff_angle;

            std::cout << "\n---------------" << std::endl;
            std::cout << "init diff angle: " << ori_diff_angle << std::endl;
            std::cout << "improved angle: " << improved_angle << std::endl;
            std::cout << "match.template_id: " << match.template_id << std::endl;
            std::cout << "match.similarity: " << match.similarity << std::endl;
        }

        std::cout << "test end" << std::endl << std::endl;
    }
}

void MIPP_test(){
    std::cout << "MIPP tests" << std::endl;
    std::cout << "----------" << std::endl << std::endl;

    std::cout << "Instr. type:       " << mipp::InstructionType                  << std::endl;
    std::cout << "Instr. full type:  " << mipp::InstructionFullType              << std::endl;
    std::cout << "Instr. version:    " << mipp::InstructionVersion               << std::endl;
    std::cout << "Instr. size:       " << mipp::RegisterSizeBit       << " bits" << std::endl;
    std::cout << "Instr. lanes:      " << mipp::Lanes                            << std::endl;
    std::cout << "64-bit support:    " << (mipp::Support64Bit    ? "yes" : "no") << std::endl;
    std::cout << "Byte/word support: " << (mipp::SupportByteWord ? "yes" : "no") << std::endl;

#ifndef has_max_int8_t
        std::cout << "in this SIMD, int8 max is not inplemented by MIPP" << std::endl;
#endif

#ifndef has_shuff_int8_t
        std::cout << "in this SIMD, int8 shuff is not inplemented by MIPP" << std::endl;
#endif

    std::cout << "----------" << std::endl << std::endl;
}

int main(){
    angle_test("test", true); // test or train
    return 0;
}
