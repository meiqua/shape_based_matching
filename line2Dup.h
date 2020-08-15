#ifndef CXXLINEMOD_H
#define CXXLINEMOD_H
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <map>
#include <functional>
#include "mipp.h"  // for SIMD in different platforms

#include <chrono>
class Timer
{
public:
    Timer() : beg_(clock_::now()) {}
    void reset() { beg_ = clock_::now(); }
    double elapsed() const
    {
        return std::chrono::duration_cast<second_>(clock_::now() - beg_).count();
    }
    double out(std::string message = "")
    {
        double t = elapsed();
        std::cout << message << "\nelasped time:" << t << "s\n"
                  << std::endl;
        reset();
        return t;
    }
    void record(std::string message = "")
    {
        if (str_time_map.find(message) == str_time_map.end())
        {
            str_time_map[message] = elapsed();
        }
        else
        {
            str_time_map[message] += elapsed();
        }
        reset();
    }
    void display(std::string message = "")
    {
        if (message == "")
        {
            for (auto item : str_time_map)
            {
                std::cout << item.first << "\nelasped time:" << item.second << "s\n"
                          << std::endl;
            }
        }
        else
        {
            std::cout << message << "\nelasped time:" << str_time_map[message] << "s\n"
                      << std::endl;
        }
    }

private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1>> second_;
    std::chrono::time_point<clock_> beg_;

    std::map<std::string, double> str_time_map;
};

class ScopeTimer: public Timer{
    ScopeTimer(std::string out_str_): out_str(out_str_){}
    ~ScopeTimer(){out(out_str);}
    std::string out_str;
};

namespace line2Dup
{

struct Feature
{
    int x;
    int y;
    int label;
    float theta;

    void read(const cv::FileNode &fn);
    void write(cv::FileStorage &fs) const;

    Feature() : x(0), y(0), label(0) {}
    Feature(int x, int y, int label);
};
inline Feature::Feature(int _x, int _y, int _label) : x(_x), y(_y), label(_label) {}

struct Template
{
    int width;
    int height;
    int tl_x;
    int tl_y;
    int pyramid_level;
    std::vector<Feature> features;

    void read(const cv::FileNode &fn);
    void write(cv::FileStorage &fs) const;
};

class ColorGradientPyramid
{
public:
    ColorGradientPyramid(const cv::Mat &src, const cv::Mat &mask,
                                             float weak_threshold, size_t num_features,
                                             float strong_threshold);

    void quantize(cv::Mat &dst) const;

    bool extractTemplate(Template &templ) const;

    void pyrDown();

public:
    void update();
    /// Candidate feature with a score
    struct Candidate
    {
        Candidate(int x, int y, int label, float score);

        /// Sort candidates with high score to the front
        bool operator<(const Candidate &rhs) const
        {
            return score > rhs.score;
        }

        Feature f;
        float score;
    };

    cv::Mat src;
    cv::Mat mask;

    int pyramid_level;
    cv::Mat angle;
    cv::Mat magnitude;
    cv::Mat angle_ori;

    float weak_threshold;
    size_t num_features;
    float strong_threshold;
    static bool selectScatteredFeatures(const std::vector<Candidate> &candidates,
                                                                            std::vector<Feature> &features,
                                                                            size_t num_features, float distance);
};
inline ColorGradientPyramid::Candidate::Candidate(int x, int y, int label, float _score) : f(x, y, label), score(_score) {}

class ColorGradient
{
public:
    ColorGradient();
    ColorGradient(float weak_threshold, size_t num_features, float strong_threshold);

    std::string name() const;

    float weak_threshold;
    size_t num_features;
    float strong_threshold;
    void read(const cv::FileNode &fn);
    void write(cv::FileStorage &fs) const;

    cv::Ptr<ColorGradientPyramid> process(const cv::Mat src, const cv::Mat &mask = cv::Mat()) const
    {
        return cv::makePtr<ColorGradientPyramid>(src, mask, weak_threshold, num_features, strong_threshold);
    }
};

struct Match
{
    Match()
    {
    }

    Match(int x, int y, float similarity, const std::string &class_id, int template_id);

    /// Sort matches with high similarity to the front
    bool operator<(const Match &rhs) const
    {
        // Secondarily sort on template_id for the sake of duplicate removal
        if (similarity != rhs.similarity)
            return similarity > rhs.similarity;
        else
            return template_id < rhs.template_id;
    }

    bool operator==(const Match &rhs) const
    {
        return x == rhs.x && y == rhs.y && similarity == rhs.similarity && class_id == rhs.class_id;
    }

    int x;
    int y;
    float similarity;
    std::string class_id;
    int template_id;
};

inline Match::Match(int _x, int _y, float _similarity, const std::string &_class_id, int _template_id)
        : x(_x), y(_y), similarity(_similarity), class_id(_class_id), template_id(_template_id)
{
}

struct FilterNode
{
    std::vector<cv::Mat> buffers;

    int num_buf = 1;
    int buffer_rows = 0;
    int buffer_cols = 0;
    int padded_rows = 0;
    int padded_cols = 0;

    int anchor_row = 0;  // anchor: where topleft is in full img
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
        r -= anchor_row;  // from full img to buffer img
        c -= anchor_col;
//        assert(r >= 0 && c >= 0);
//        r = r % buffer_rows;  // row is changed because of rolling buffer
        return &buffers[buf_idx].at<T>(r, c);
    }

    std::function<int(int, int, int, int)> simple_update;  // update start_r end_r start_c end_c
    std::function<int(int, int, int, int)> simd_update;

    void backward_rc(std::vector<FilterNode>& nodes, int rows, int cols, int cur_padded_rows, int cur_padded_cols) // calculate paddings
    {
        if (rows > buffer_rows){
            buffer_rows = rows;
            padded_rows = cur_padded_rows;
        }
        if (cols > buffer_cols){
            buffer_cols = cols;
            padded_cols = cur_padded_cols;
        }
        if (parent >= 0)
            nodes[parent].backward_rc(nodes, buffer_rows+op_r - 1, cols+op_c - 1,
                                      cur_padded_rows+op_r/2, cur_padded_cols+op_c/2);
    }
};

class Detector
{
public:
    /**
         * \brief Empty constructor, initialize with read().
         */
    Detector();

    Detector(std::vector<int> T);
    Detector(int num_features, std::vector<int> T, float weak_thresh = 30.0f, float strong_thresh = 60.0f);

    std::vector<Match> match(cv::Mat sources, float threshold,
                                                     const std::vector<std::string> &class_ids = std::vector<std::string>(),
                                                     const cv::Mat masks = cv::Mat());
    int addTemplate(const cv::Mat sources, const std::string &class_id,
                                    const cv::Mat &object_mask, int num_features = 0);

    int addTemplate_rotate(const std::string &class_id, int zero_id, float theta, cv::Point2f center);

    const cv::Ptr<ColorGradient> &getModalities() const { return modality; }

    int getT(int pyramid_level) const { return T_at_level[pyramid_level]; }

    int pyramidLevels() const { return pyramid_levels; }

    const std::vector<Template> &getTemplates(const std::string &class_id, int template_id) const;

    int numTemplates() const;
    int numTemplates(const std::string &class_id) const;
    int numClasses() const { return static_cast<int>(class_templates.size()); }

    std::vector<std::string> classIds() const;

    void read(const cv::FileNode &fn);
    void write(cv::FileStorage &fs) const;

    std::string readClass(const cv::FileNode &fn, const std::string &class_id_override = "");
    void writeClass(const std::string &class_id, cv::FileStorage &fs) const;

    void readClasses(const std::vector<std::string> &class_ids,
                                     const std::string &format = "templates_%s.yml.gz");
    void writeClasses(const std::string &format = "templates_%s.yml.gz") const;

    bool set_produce_dxy = false;
    cv::Mat dx_, dy_;
protected:
    cv::Ptr<ColorGradient> modality;
    int pyramid_levels;
    std::vector<int> T_at_level;

    float res_map_mag_thresh = 60.0f;

    typedef std::vector<Template> TemplatePyramid;
    typedef std::map<std::string, std::vector<TemplatePyramid>> TemplatesMap;
    TemplatesMap class_templates;

    typedef std::vector<cv::Mat> LinearMemories;
    // Indexed as [pyramid level][ColorGradient][quantized label]
    typedef std::vector<std::vector<LinearMemories>> LinearMemoryPyramid;

    void matchClass(const LinearMemoryPyramid &lm_pyramid,
                                    const std::vector<cv::Size> &sizes,
                                    float threshold, std::vector<Match> &matches,
                                    const std::string &class_id,
                                    const std::vector<TemplatePyramid> &template_pyramids) const;
};

} // namespace line2Dup

namespace shape_based_matching {
class shapeInfo_producer{
public:
    cv::Mat src;
    cv::Mat mask;

    std::vector<float> angle_range;
    std::vector<float> scale_range;

    float angle_step = 15;
    float scale_step = 0.5;
    float eps = 0.00001f;

    class Info{
    public:
        float angle;
        float scale;

        Info(float angle_, float scale_){
            angle = angle_;
            scale = scale_;
        }
    };
    std::vector<Info> infos;

    shapeInfo_producer(cv::Mat src, cv::Mat mask = cv::Mat()){
        this->src = src;
        if(mask.empty()){
            // make sure we have masks
            this->mask = cv::Mat(src.size(), CV_8UC1, {255});
        }else{
            this->mask = mask;
        }
    }

    static cv::Mat transform(cv::Mat src, float angle, float scale){
        cv::Mat dst;

        cv::Point2f center(src.cols/2.0f, src.rows/2.0f);
        cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, scale);
        cv::warpAffine(src, dst, rot_mat, src.size());

        return dst;
    }
    static void save_infos(std::vector<shapeInfo_producer::Info>& infos, std::string path = "infos.yaml"){
        cv::FileStorage fs(path, cv::FileStorage::WRITE);

        fs << "infos"
           << "[";
        for (int i = 0; i < infos.size(); i++)
        {
            fs << "{";
            fs << "angle" << infos[i].angle;
            fs << "scale" << infos[i].scale;
            fs << "}";
        }
        fs << "]";
    }
    static std::vector<Info> load_infos(std::string path = "info.yaml"){
        cv::FileStorage fs(path, cv::FileStorage::READ);

        std::vector<Info> infos;

        cv::FileNode infos_fn = fs["infos"];
        cv::FileNodeIterator it = infos_fn.begin(), it_end = infos_fn.end();
        for (int i = 0; it != it_end; ++it, i++)
        {
            infos.emplace_back(float((*it)["angle"]), float((*it)["scale"]));
        }
        return infos;
    }

    void produce_infos(){
        infos.clear();

        assert(angle_range.size() <= 2);
        assert(scale_range.size() <= 2);
        assert(angle_step > eps*10);
        assert(scale_step > eps*10);

        // make sure range not empty
        if(angle_range.size() == 0){
            angle_range.push_back(0);
        }
        if(scale_range.size() == 0){
            scale_range.push_back(1);
        }

        if(angle_range.size() == 1 && scale_range.size() == 1){
            float angle = angle_range[0];
            float scale = scale_range[0];
            infos.emplace_back(angle, scale);

        }else if(angle_range.size() == 1 && scale_range.size() == 2){
            assert(scale_range[1] > scale_range[0]);
            float angle = angle_range[0];
            for(float scale = scale_range[0]; scale <= scale_range[1]+eps; scale += scale_step){
                infos.emplace_back(angle, scale);
            }
        }else if(angle_range.size() == 2 && scale_range.size() == 1){
            assert(angle_range[1] > angle_range[0]);
            float scale = scale_range[0];
            for(float angle = angle_range[0]; angle <= angle_range[1]+eps; angle += angle_step){
                infos.emplace_back(angle, scale);
            }
        }else if(angle_range.size() == 2 && scale_range.size() == 2){
            assert(scale_range[1] > scale_range[0]);
            assert(angle_range[1] > angle_range[0]);
            for(float scale = scale_range[0]; scale <= scale_range[1]+eps; scale += scale_step){
                for(float angle = angle_range[0]; angle <= angle_range[1]+eps; angle += angle_step){
                    infos.emplace_back(angle, scale);
                }
            }
        }
    }

    cv::Mat src_of(const Info& info){
        return transform(src, info.angle, info.scale);
    }

    cv::Mat mask_of(const Info& info){
        return (transform(mask, info.angle, info.scale) > 0);
    }
};

}

#endif
