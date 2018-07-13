#include "line2Dup.h"
#include <memory>
#include <iostream>
#include <assert.h>
#include <chrono>

using namespace std;
using namespace cv;

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

static std::string prefix = "/home/meiqua/shape_based_matching/test/";

void circle_gen(){
    Mat bg = Mat(200, 200, CV_8UC3, {0, 0, 0});
    cv::circle(bg, {100, 100}, 50, {255,255,255}, -1);
    cv::imshow("test", bg);
    waitKey(0);
}

int main(){
    line2Dup::Detector detector(128, {2, 8});

    string mode = "train";
    mode = "test";
    if(mode == "train"){
        Mat img = cv::imread(prefix+"templ/circle.png");
        shape_based_matching::shapeInfo shapes(img);
        shapes.scale_range = {0.2f, 2};
        shapes.scale_step = 0.05f;
        shapes.produce_infos();
        std::vector<shape_based_matching::shapeInfo::shape_and_info> infos_have_templ;
        string class_id = "circle";
        for(auto& info: shapes.infos){
            int templ_id = detector.addTemplate(info.src, class_id, info.mask);
            std::cout << "templ_id: " << templ_id << std::endl;
            if(templ_id != -1){
                infos_have_templ.push_back(info);
            }
        }
        detector.writeClasses(prefix+"%s_templ.yaml");
        shapes.save_infos(infos_have_templ, shapes.src, shapes.mask, prefix + "circle_info.yaml");
        std::cout << "train end" << std::endl;
    }else if(mode=="test"){
        std::vector<std::string> ids;
        ids.push_back("circle");
        detector.readClasses(ids, prefix+"%s_templ.yaml");

        Mat test_img = imread(prefix+"t4.png");
        Rect roi(0, 0, 1024 ,1024);
        Mat img = test_img(roi).clone();
        assert(img.isContinuous());

        Timer timer;
        auto matches = detector.match(img, 75, ids);
        timer.out();

        std::cout << "matches.size(): " << matches.size() << std::endl;
        size_t top5 = 5;
        if(top5>matches.size()) top5=matches.size();
        for(size_t i=0; i<top5; i++){
            auto match = matches[i];
            auto templ = detector.getTemplates("circle",
                                               match.template_id);
            int x =  templ[0].width/2 + match.x;
            int y = templ[0].height/2 + match.y;
            int r = templ[0].width/2;
            Scalar color(255, rand()%255, rand()%255);

            cv::putText(img, to_string(int(round(match.similarity))),
                        Point(match.x+r-10, match.y-3), FONT_HERSHEY_PLAIN, 2, color);
            cv::circle(img, {x, y}, r, color, 2);
        }

        imshow("img", img);
        waitKey(0);

        std::cout << "test end" << std::endl;
    }
    return 0;
}
