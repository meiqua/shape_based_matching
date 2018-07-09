#ifndef CXXLINEMOD_H
#define CXXLINEMOD_H
#include <opencv2/core/core.hpp>
#include <opencv2/rgbd.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace line2Dup
{

struct Feature
{
    int x;
    int y;
    int label;

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

protected:
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

    cv::Ptr<ColorGradientPyramid> process(const std::vector<cv::Mat> &src,
                                                                                const cv::Mat &mask = cv::Mat()) const
    {
        return processImpl(src, mask);
    }

    cv::Ptr<ColorGradientPyramid> processImpl(const std::vector<cv::Mat> &src,
                                                                                        const cv::Mat &mask) const;
    static cv::Ptr<ColorGradient> create(const std::string &ColorGradient_type);
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

class Detector
{
public:
    /**
         * \brief Empty constructor, initialize with read().
         */
    Detector();

    Detector(std::vector<int> T);
    Detector(int num_features, std::vector<int> T);

    std::vector<Match> match(cv::Mat sources, float threshold,
                                                     const std::vector<std::string> &class_ids = std::vector<std::string>(),
                                                     const cv::Mat masks = cv::Mat()) const;

    int addTemplate(const std::vector<cv::Mat> &sources, const std::string &class_id,
                                    const cv::Mat &object_mask);

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

protected:
    cv::Ptr<ColorGradient> modality;
    int pyramid_levels;
    std::vector<int> T_at_level;

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

#endif
