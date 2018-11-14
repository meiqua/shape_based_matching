#include "line2Dup.h"
#include <memory>
#include <iostream>
#include <assert.h>
#include <chrono>
#include <opencv2/dnn.hpp>
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
    Mat bg = Mat(800, 800, CV_8UC3, {0, 0, 0});
    cv::circle(bg, {400, 400}, 200, {255,255,255}, -1);
    cv::imshow("test", bg);
    waitKey(0);
}

void scale_test(){
    int num_feature = 150;
    line2Dup::Detector detector(num_feature, {4, 8});

    string mode = "train";
    mode = "test";
    if(mode == "train"){
        Mat img = cv::imread(prefix+"case0/templ/circle.png");
        shape_based_matching::shapeInfo shapes(img);
        shapes.scale_range = {0.1f, 1};
        shapes.scale_step = 0.01f;
        shapes.produce_infos();
        std::vector<shape_based_matching::shapeInfo::shape_and_info> infos_have_templ;
        string class_id = "circle";
        for(auto& info: shapes.infos){
            int templ_id = detector.addTemplate(info.src, class_id, info.mask,
                                                int(num_feature*info.scale));
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

        Mat test_img = imread(prefix+"case0/3.png");

        int stride = 32;
        int n = test_img.rows/stride;
        int m = test_img.cols/stride;
        Rect roi(0, 0, stride*m , stride*n);
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

//            int cols = templ[0].width + 1;
//            int rows = templ[0].height+ 1;
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

        std::cout << "test img size: " << img.rows * img.cols << std::endl;

        Timer timer;
        auto matches = detector.match(img, 90, ids);
        timer.out();

        if(img.channels() == 1) cvtColor(img, img, CV_GRAY2BGR);

        std::cout << "matches.size(): " << matches.size() << std::endl;
        size_t top5 = 50;
        if(top5>matches.size()) top5=matches.size();
        for(size_t i=0; i<top5; i++){
            auto match = matches[i];
            auto templ = detector.getTemplates("test",
                                               match.template_id);

//            int cols = templ[0].width + 1;
//            int rows = templ[0].height+ 1;
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

void noise_test(){
    line2Dup::Detector detector(30, {4, 8});

    string mode = "train";
    mode = "test";
//    mode = "none";
    if(mode == "train"){
        Mat img = imread(prefix+"case2/train.png");
        Mat mask = Mat(img.size(), CV_8UC1, {255});

        shape_based_matching::shapeInfo shapes(img, mask);
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
        detector.writeClasses(prefix+"case2/%s_templ.yaml");
        shapes.save_infos(infos_have_templ, shapes.src, shapes.mask, prefix + "case2/test_info.yaml");
        std::cout << "train end" << std::endl;
    }else if(mode=="test"){
        std::vector<std::string> ids;
        ids.push_back("test");
        detector.readClasses(ids, prefix+"case2/%s_templ.yaml");

        Mat test_img = imread(prefix+"case2/test.png");

        int stride = 16;
        int n = test_img.rows/stride;
        int m = test_img.cols/stride;
        Rect roi(0, 0, stride*m , stride*n);

        test_img = test_img(roi).clone();

        Timer timer;
        auto matches = detector.match(test_img, 90, ids);
        timer.out();

        std::cout << "matches.size(): " << matches.size() << std::endl;
        size_t top5 = 500;
        if(top5>matches.size()) top5=matches.size();

        vector<Rect> boxes;
        vector<float> scores;
        vector<int> idxs;
        for(auto match: matches){
            Rect box;
            box.x = match.x;
            box.y = match.y;

            auto templ = detector.getTemplates("test",
                                               match.template_id);

            box.width = templ[0].width;
            box.height = templ[0].height;
            boxes.push_back(box);
            scores.push_back(match.similarity);
        }
        cv::dnn::NMSBoxes(boxes, scores, 0, 0.5f, idxs);

        for(auto idx: idxs){
            auto match = matches[idx];
            auto templ = detector.getTemplates("test",
                                               match.template_id);

            int x =  templ[0].width + match.x;
            int y = templ[0].height + match.y;
            int r = templ[0].width/2;
            cv::Vec3b randColor;
            randColor[0] = rand()%155 + 100;
            randColor[1] = rand()%155 + 100;
            randColor[2] = rand()%155 + 100;

            for(int i=0; i<templ[0].features.size(); i++){
                auto feat = templ[0].features[i];
                cv::circle(test_img, {feat.x+match.x, feat.y+match.y}, 2, randColor, -1);
            }

            cv::putText(test_img, to_string(int(round(match.similarity))),
                        Point(match.x+r-10, match.y-3), FONT_HERSHEY_PLAIN, 2, randColor);
            cv::rectangle(test_img, {match.x, match.y}, {x, y}, randColor, 2);

            std::cout << "\nmatch.template_id: " << match.template_id << std::endl;
            std::cout << "match.similarity: " << match.similarity << std::endl;
        }

        imshow("img", test_img);
        waitKey(0);

        std::cout << "test end" << std::endl;
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
    std::cout << "----------" << std::endl << std::endl;
}

int main(){

    MIPP_test();
//    angle_test();
    scale_test();
    noise_test();
    return 0;
}
