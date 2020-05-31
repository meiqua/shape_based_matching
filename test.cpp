#include "line2Dup.h"
#include <memory>
#include <iostream>
#include <assert.h>
#include <chrono>
using namespace std;
using namespace cv;

static std::string prefix = "/home/meiqua/shape_based_matching/test/";

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

void angle_test(string mode = "test", bool use_rot = true){
    line2Dup::Detector detector(128, {4, 8});

//    mode = "test";
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
        shapes.produce_infos();
        std::vector<shape_based_matching::shapeInfo_producer::Info> infos_have_templ;
        string class_id = "test";

        bool is_fisrt = true;
        int first_id = 0;
        float first_angle = 0;
        for(auto& info: shapes.infos){
            Mat to_show = shapes.src_of(info);

            std::cout << "\ninfo.angle: " << info.angle << std::endl;
            int templ_id;

            if(is_fisrt){
                templ_id = detector.addTemplate(shapes.src_of(info), class_id, shapes.mask_of(info));
                first_id = templ_id;
                first_angle = info.angle;

                if(use_rot) is_fisrt = false;
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

        Mat test_img = imread(prefix+"case1/test.png");
        assert(!test_img.empty() && "check your img path");

        int padding = 250;
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
        auto matches = detector.match(img, 50, ids);
        timer.out();

        if(img.channels() == 1) cvtColor(img, img, CV_GRAY2BGR);

        std::cout << "matches.size(): " << matches.size() << std::endl;
        size_t top5 = 1;
        if(top5>matches.size()) top5=matches.size();
        for(size_t i=0; i<top5; i++){
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

            // center x,y of train_img in test img
            float x =  match.x - templ[0].tl_x + train_img_half_width;
            float y =  match.y - templ[0].tl_y + train_img_half_width;

            cv::Vec3b randColor;
            randColor[0] = rand()%155 + 100;
            randColor[1] = rand()%155 + 100;
            randColor[2] = rand()%155 + 100;
            for(int i=0; i<templ[0].features.size(); i++){
                auto feat = templ[0].features[i];
                cv::circle(img, {feat.x+match.x, feat.y+match.y}, 3, randColor, -1);
            }

            cv::putText(img, to_string(int(round(match.similarity))),
                        Point(match.x+r_scaled-10, match.y-3), FONT_HERSHEY_PLAIN, 2, randColor);

            cv::RotatedRect rotatedRectangle({x, y}, {2*r_scaled, 2*r_scaled}, -infos[match.template_id].angle);

            cv::Point2f vertices[4];
            rotatedRectangle.points(vertices);
            for(int i=0; i<4; i++){
                int next = (i+1==4) ? 0 : (i+1);
                cv::line(img, vertices[i], vertices[next], randColor, 2);
            }

            std::cout << "\nmatch.template_id: " << match.template_id << std::endl;
            std::cout << "match.similarity: " << match.similarity << std::endl;
        }

        imshow("img", img);
        waitKey(0);

        std::cout << "test end" << std::endl << std::endl;
    }
}


int main(){

    angle_test("test", true); // test or train
    return 0;
}
