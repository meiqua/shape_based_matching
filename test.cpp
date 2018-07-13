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

void scale_test(){
    line2Dup::Detector detector(128, {4, 8});

    string mode = "train";
    mode = "test";
    if(mode == "train"){
        Mat img = cv::imread(prefix+"case0/templ/circle.png");
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
        detector.writeClasses(prefix+"case0/%s_templ.yaml");
        shapes.save_infos(infos_have_templ, shapes.src, shapes.mask, prefix + "circle_info.yaml");
        std::cout << "train end" << std::endl;

    }else if(mode=="test"){
        std::vector<std::string> ids;
        ids.push_back("circle");
        detector.readClasses(ids, prefix+"case0/%s_templ.yaml");

        Mat test_img = imread(prefix+"case0/t1.png");
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
}

void angle_test(){
    line2Dup::Detector detector(128, {4, 8});

    string mode = "train";
    mode = "test";
//    mode = "none";
    if(mode == "train"){
        Mat img = imread(prefix+"case1/train.png");
        Rect roi(130, 110, 270, 270);
        img = img(roi).clone();
        Mat mask = Mat(img.size(), CV_8UC1, {255});

        // padding to avoid rotating out
        int padding = 100;
        cv::Mat padded_img = cv::Mat(img.rows + 2*padding, img.cols + 2*padding, img.type(), cv::Scalar::all(0));
        img.copyTo(padded_img(Rect(padding, padding, img.cols, img.rows)));

        cv::Mat padded_mask = cv::Mat(mask.rows + 2*padding, mask.cols + 2*padding, mask.type(), cv::Scalar::all(0));
        mask.copyTo(padded_mask(Rect(padding, padding, img.cols, img.rows)));

        shape_based_matching::shapeInfo shapes(padded_img, padded_mask);
        shapes.angle_range = {0, 360};
        shapes.angle_step = 1;
        shapes.produce_infos();
        std::vector<shape_based_matching::shapeInfo::shape_and_info> infos_have_templ;
        string class_id = "test";
        for(auto& info: shapes.infos){
            imshow("train", info.src);
            waitKey(1);

            std::cout << "\ninfo.angle: " << info.angle << std::endl;
            int templ_id = detector.addTemplate(info.src, class_id, info.mask);
            std::cout << "templ_id: " << templ_id << std::endl;
            if(templ_id != -1){
                infos_have_templ.push_back(info);
            }
        }
        detector.writeClasses(prefix+"case1/%s_templ.yaml");
        shapes.save_infos(infos_have_templ, shapes.src, shapes.mask, prefix + "case1/test_info.yaml");
        std::cout << "train end" << std::endl;
    }else if(mode=="test"){
        std::vector<std::string> ids;
        ids.push_back("test");
        detector.readClasses(ids, prefix+"case1/%s_templ.yaml");

        Mat test_img = imread(prefix+"case1/test.png");

        int padding = 500;
        cv::Mat padded_img = cv::Mat(test_img.rows + 2*padding,
                                     test_img.cols + 2*padding, test_img.type(), cv::Scalar::all(0));
        test_img.copyTo(padded_img(Rect(padding, padding, test_img.cols, test_img.rows)));

        Rect roi(200, 200, 1024 ,1024);
        Mat img = padded_img(roi).clone();
        assert(img.isContinuous());

        imshow("test", img);
        waitKey(0);

        Timer timer;
        auto matches = detector.match(img, 90, ids);
        timer.out();

        std::cout << "matches.size(): " << matches.size() << std::endl;
        size_t top5 = 5;
        if(top5>matches.size()) top5=matches.size();
        for(size_t i=0; i<top5; i++){
            auto match = matches[i];
            auto templ = detector.getTemplates("test",
                                               match.template_id);

//            int cols = templ[0].width + templ[0].tl_x;
//            int rows = templ[0].height+ templ[0].tl_y;
//            cv::Mat view = cv::Mat(rows, cols, CV_8UC1, cv::Scalar(0));
//            for(int i=0; i<templ[0].features.size(); i++){
//                auto feat = templ[0].features[i];
//                assert(feat.y<rows);
//                assert(feat.x<cols);
//                view.at<uchar>(feat.y, feat.x) = 255;
//            }
//            view = view>0;
//            imshow("test", view);
//            waitKey(0);

            int x =  templ[0].width/2 + match.x;
            int y = templ[0].height/2 + match.y;
            int r = templ[0].width/2;
            Scalar color(255, rand()%255, rand()%255);

            cv::putText(img, to_string(int(round(match.similarity))),
                        Point(match.x+r-10, match.y-3), FONT_HERSHEY_PLAIN, 2, color);
            cv::circle(img, {x, y}, r, color, 2);

            std::cout << "\nmatch.template_id: " << match.template_id << std::endl;
            std::cout << "match.similarity: " << match.similarity << std::endl;
        }

        imshow("img", img);
        waitKey(0);

        std::cout << "test end" << std::endl;
    }
}
int main(){
    scale_test();
    return 0;
}
