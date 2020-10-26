#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <map>
#include <memory>
#include <iostream>
#include <assert.h>
#include <chrono>
#include <stdlib.h>

#include "mipp.h"  // for SIMD in different platforms

namespace simple_fusion {

#define INVALID 63 // > 8 && < 128

inline int CvTypeSize(int type){
    if(type == CV_8U) return 1;
    else if(type == CV_8UC3) return 3;
    else if(type == CV_16U) return 2;
    else if(type == CV_16S) return 2;
    else if(type == CV_32S) return 4;
    else if(type == CV_32F) return 4;
    else{
        CV_Error(cv::Error::StsBadArg, "Invalid type");
    }
    return 1;
}

inline int aligned256_after_n_char(const char* src){
    int c = 0;
    while (reinterpret_cast<unsigned long long>(src + c) % 32 != 0){
        c++;
    }
    return c;
}

template<typename T>
std::vector<std::shared_ptr<T>> deep_copy_shared_ptr_vec(std::vector<std::shared_ptr<T>>& ori){
    std::vector<std::shared_ptr<T>> res;
    for(auto& ori_ptr: ori) res.push_back(ori_ptr->clone());
    return res;
}

class FilterNode
{
public:
    FilterNode(std::string op_name_temp, int input_type_temp,
               int input_num_temp, int output_type_temp, int output_num_temp,
               int op_row_temp, int op_col_temp): op_name(op_name_temp),
    input_type(input_type_temp), input_num(input_num_temp), output_type(output_type_temp),
    output_num(output_num_temp), op_row(op_row_temp), op_col(op_col_temp){}

    FilterNode() = delete; // delete default to force user to provide filter infos

    std::string op_name;

    int input_type = CV_16U;
    int input_num = 1;
    int output_type = CV_16U;
    int output_num = 1;

    int op_row = 3;
    int op_col = 3;
    int padded_row = 0;
    int padded_col = 0;

    // for linearize we need to know gobal index
    int cur_row = 0;
    int cur_col = 0;

    // for updates' memory address
    std::vector<cv::Mat> in_headers;
    std::vector<cv::Mat> out_headers;
    int which_buffer = 0;

    bool have_special_headers = false;
    virtual void link_special_header(const cv::Rect& cur_roi){}

    bool use_simd = true;
    virtual void update_simple(int start_r, int start_c, int end_r, int end_c) = 0;
    virtual void update_simd(int start_r, int start_c, int end_r, int end_c) = 0;
    void update(){
        if(use_simd)
            update_simd(op_row/2, op_col/2, in_headers[0].rows - op_row/2, in_headers[0].cols - op_col/2);
        else
            update_simple(op_row/2, op_col/2, in_headers[0].rows - op_row/2, in_headers[0].cols - op_col/2);
    }

    virtual std::shared_ptr<FilterNode> clone() const = 0;
};

// dummy node for test and node template
class Dummy1X1Node_8U_8U : public FilterNode {
public:
    Dummy1X1Node_8U_8U() : FilterNode("dummy", CV_8U, 1, CV_8U, 1, 1, 1) {}
    void update_simple(int start_r, int start_c, int end_r, int end_c) override {
        for(int r = start_r; r < end_r; r++){
            int c = start_c;
            uint8_t *parent_buf_ptr = in_headers[0].ptr<uint8_t>(r, c);
            uint8_t *buf_ptr = out_headers[0].ptr<uint8_t>(r - op_row/2, c - op_col/2);
            for(; c < end_c; c++, parent_buf_ptr++, buf_ptr++){
                *buf_ptr = *parent_buf_ptr;
            }
        }
    }
    void update_simd(int start_r, int start_c, int end_r, int end_c) override {
        const int simd_step = mipp::N<int8_t>();
        for(int r = start_r; r < end_r; r++){
            int c = start_c;
            uint8_t *parent_buf_ptr = in_headers[0].ptr<uint8_t>(r, c);
            uint8_t *buf_ptr = out_headers[0].ptr<uint8_t>(r - op_row/2, c - op_col/2);

            for (; c <= end_c - simd_step; c += simd_step, buf_ptr += simd_step, parent_buf_ptr += simd_step){
                mipp::Reg<uint8_t> src8_v0(parent_buf_ptr);
                src8_v0.store(buf_ptr);
            }
            for(; c < end_c; c++, parent_buf_ptr++, buf_ptr++){
                *buf_ptr = *parent_buf_ptr;
            }
        }
    }
    std::shared_ptr<FilterNode> clone() const override {
        std::shared_ptr<FilterNode> node_new = std::make_shared<Dummy1X1Node_8U_8U>();
        node_new->padded_row = padded_row;
        node_new->padded_col = padded_col;
        node_new->which_buffer = which_buffer;
        return node_new;
    }
};

class BGR2GRAY_8UC3_8U : public FilterNode {
public:
    BGR2GRAY_8UC3_8U() : FilterNode("bgr2gray", CV_8UC3, 1, CV_8U, 1, 1, 1) {}
    void update_simple(int start_r, int start_c, int end_r, int end_c) override {
        // assert(in_headers[0].rows == out_headers[0].rows
        //     && in_headers[0].cols == out_headers[0].cols
        //     && "we will use opencv directly, r/c should be same");

        cv::Mat gray;
        // opencv can't specify an existing one?
        // cv::COLOR_BGR2GRAY is new version of CV_BGR2GRAY
        cv::cvtColor(in_headers[0], gray, cv::COLOR_BGR2GRAY);
        gray.copyTo(out_headers[0]);
    }
    void update_simd(int start_r, int start_c, int end_r, int end_c) override {
        update_simple(start_r, start_c, end_r, end_c);
    }
    std::shared_ptr<FilterNode> clone() const override {
        std::shared_ptr<FilterNode> node_new = std::make_shared<BGR2GRAY_8UC3_8U>();
        node_new->padded_row = padded_row;
        node_new->padded_col = padded_col;
        node_new->which_buffer = which_buffer;
        return node_new;
    }
};


class Gauss1x5Node_8U_32S_4bit_larger: public FilterNode {
public:
    Gauss1x5Node_8U_32S_4bit_larger(): FilterNode("gauss1x5", CV_8U, 1, CV_32S, 1, 1, 5){
        cv::Mat double_gauss = cv::getGaussianKernel(gauss_size, 0, CV_64F);
        for(int i=0; i<gauss_size; i++){
            gauss_knl_uint32[i] = int32_t(double_gauss.at<double>(i, 0) * (1<<gauss_quant_bit));
        }
    }
    void update_simple(int start_r, int start_c, int end_r, int end_c) override {
        for(int r = start_r; r < end_r; r++){
            int c = start_c;
            uint8_t *parent_buf_ptr = in_headers[0].ptr<uint8_t>(r, c);
            int32_t *buf_ptr = out_headers[0].ptr<int32_t>(r - op_row/2, c - op_col/2);
            for(; c < end_c; c++, parent_buf_ptr++, buf_ptr++){
                int32_t local_sum = gauss_knl_uint32[gauss_size/2]*int32_t(*(parent_buf_ptr));
                int gauss_knl_idx = gauss_size/2 + 1;
                for(int i=1; i<=gauss_size/2; i++, gauss_knl_idx++){
                    local_sum += (int32_t(*(parent_buf_ptr + i)) +
                                  int32_t(*(parent_buf_ptr - i))) * gauss_knl_uint32[gauss_knl_idx];
                }
                *buf_ptr = local_sum;
            }
        }
    }
    void update_simd(int start_r, int start_c, int end_r, int end_c) override {
        const int simd_step = mipp::N<int32_t>();
        const mipp::Reg<uint8_t> zero8_v = uint8_t(0);
        const mipp::Reg<int16_t> zero16_v = int16_t(0);
        for(int r = start_r; r < end_r; r++){
            int c = start_c;
            uint8_t *parent_buf_ptr = in_headers[0].ptr<uint8_t>(r, c);
            int32_t *buf_ptr = out_headers[0].ptr<int32_t>(r - op_row/2, c - op_col/2);

            // 4* because parent read 4x long, we want to avoid reading out of tile
            for (; c <= end_c - 4*simd_step; c += simd_step, buf_ptr += simd_step, parent_buf_ptr += simd_step){
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
            for(; c < end_c; c++, parent_buf_ptr++, buf_ptr++){
                int32_t local_sum = gauss_knl_uint32[gauss_size/2]*int32_t(*(parent_buf_ptr));
                int gauss_knl_idx = gauss_size/2 + 1;
                for(int i=1; i<=gauss_size/2; i++, gauss_knl_idx++){
                    local_sum += (int32_t(*(parent_buf_ptr + i)) +
                                  int32_t(*(parent_buf_ptr - i))) * gauss_knl_uint32[gauss_knl_idx];
                }
                *buf_ptr = local_sum;
            }
        }
    }
    std::shared_ptr<FilterNode> clone() const override {
        std::shared_ptr<FilterNode> node_new = std::make_shared<Gauss1x5Node_8U_32S_4bit_larger>();
        node_new->padded_row = padded_row;
        node_new->padded_col = padded_col;
        node_new->which_buffer = which_buffer;
        return node_new;
    }
    const int gauss_quant_bit = 4;  // should be larger if gauss_size is larger
    const int gauss_size = 5;
    int32_t gauss_knl_uint32[5];
};

class Gauss5x1Node_32S_16S_4bit_smaller : public FilterNode {
public:
    Gauss5x1Node_32S_16S_4bit_smaller(): FilterNode("gauss5x1", CV_32S, 1, CV_16S, 1, 5, 1){
        cv::Mat double_gauss = cv::getGaussianKernel(gauss_size, 0, CV_64F);
        for(int i=0; i<gauss_size; i++){
            gauss_knl_uint32[i] = int32_t(double_gauss.at<double>(i, 0) * (1<<gauss_quant_bit));
        }
    }
    void update_simple(int start_r, int start_c, int end_r, int end_c) override {
        for(int r = start_r; r < end_r; r++){
            int c = start_c;
            int16_t *buf_ptr = out_headers[0].ptr<int16_t>(r - op_row/2, c - op_col/2);

            std::vector<int32_t*> parent_buf_ptr(gauss_size);
            int32_t** parent_buf_center = &parent_buf_ptr[gauss_size/2];
            parent_buf_center[0] = in_headers[0].ptr<int32_t>(r, c);
            for(int i=1; i<=gauss_size/2; i++){
                parent_buf_center[i] = in_headers[0].ptr<int32_t>(r+i, c);
                parent_buf_center[-i] = in_headers[0].ptr<int32_t>(r-i, c);
            }
            for (; c < end_c; c++, buf_ptr++){
                int32_t local_sum = gauss_knl_uint32[gauss_size/2] * (*(parent_buf_center[0]));
                int gauss_knl_idx = gauss_size/2 + 1;
                for(int i=1; i<=gauss_size/2; i++, gauss_knl_idx++){
                    local_sum += gauss_knl_uint32[gauss_knl_idx] *
                            (*(parent_buf_center[i]) + *(parent_buf_center[-i]));
                }
                *buf_ptr = int16_t(local_sum >> (2*gauss_quant_bit));

                for(int i=0; i<gauss_size; i++){
                    parent_buf_ptr[i]++;
                }
            }
        }
    }

    void update_simd(int start_r, int start_c, int end_r, int end_c) override {
        const int simd_step = mipp::N<int32_t>();
        const mipp::Reg<int32_t> zero32_v = int32_t(0);
        for(int r = start_r; r < end_r; r++){
            int c = start_c;
            int16_t *buf_ptr = out_headers[0].ptr<int16_t>(r - op_row/2, c - op_col/2);

            std::vector<int32_t*> parent_buf_ptr(gauss_size);
            int32_t** parent_buf_center = &parent_buf_ptr[gauss_size/2];
            parent_buf_center[0] = in_headers[0].ptr<int32_t>(r, c);
            for(int i=1; i<=gauss_size/2; i++){
                parent_buf_center[i] = in_headers[0].ptr<int32_t>(r+i, c);
                parent_buf_center[-i] = in_headers[0].ptr<int32_t>(r-i, c);
            }
            for (; c <= end_c - 2*simd_step; c += simd_step, buf_ptr += simd_step){
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

                mipp::Reg<int16_t> local_sum_int16 = mipp::pack<int32_t,int16_t>(local_sum, zero32_v);
                local_sum_int16.store(buf_ptr);

                for(int i=0; i<gauss_size; i++){
                    parent_buf_ptr[i] += simd_step;
                }
            }
            for (; c < end_c; c++, buf_ptr++){
                int32_t local_sum = gauss_knl_uint32[gauss_size/2] * (*(parent_buf_center[0]));
                int gauss_knl_idx = gauss_size/2 + 1;
                for(int i=1; i<=gauss_size/2; i++, gauss_knl_idx++){
                    local_sum += gauss_knl_uint32[gauss_knl_idx] *
                            (*(parent_buf_center[i]) + *(parent_buf_center[-i]));
                }
                *buf_ptr = int16_t(local_sum >> (2*gauss_quant_bit));

                for(int i=0; i<gauss_size; i++){
                    parent_buf_ptr[i]++;
                }
            }
        }
    }
    std::shared_ptr<FilterNode> clone() const override {
        std::shared_ptr<FilterNode> node_new = std::make_shared<Gauss5x1Node_32S_16S_4bit_smaller>();
        node_new->padded_row = padded_row;
        node_new->padded_col = padded_col;
        node_new->which_buffer = which_buffer;
        return node_new;
    }
    const int gauss_quant_bit = 4; // should be larger if gauss_size is larger
    const int gauss_size = 5;
    int32_t gauss_knl_uint32[5];
};

class Gauss5x1withPyrdownNode_32S_16S_4bit_smaller : public FilterNode {
public:
    Gauss5x1withPyrdownNode_32S_16S_4bit_smaller(cv::Mat down_img, bool need_pyr = true):
        down_img_(down_img), need_pyr_(need_pyr), FilterNode("gauss5x1", CV_32S, 1, CV_16S, 1, 5, 1){
        cv::Mat double_gauss = cv::getGaussianKernel(gauss_size, 0, CV_64F);
        for(int i=0; i<gauss_size; i++){
            gauss_knl_uint32[i] = int32_t(double_gauss.at<double>(i, 0) * (1<<gauss_quant_bit));
        }

        if(need_pyr_){
            have_special_headers = true;
        }
    }
    void update_simple(int start_r, int start_c, int end_r, int end_c) override {
        for(int r = start_r; r < end_r; r++){
            bool is_even_row = r % 2;
            int r_down = r / 2 - padded_row / 2;
            bool is_r_in_down_roi;
            if(need_pyr_){
                is_r_in_down_roi = (r_down >= 0 && r_down < out_headers[1].rows);
            }
            int c = start_c;
            int16_t *buf_ptr = out_headers[0].ptr<int16_t>(r - op_row/2, c - op_col/2);

            std::vector<int32_t*> parent_buf_ptr(gauss_size);
            int32_t** parent_buf_center = &parent_buf_ptr[gauss_size/2];
            parent_buf_center[0] = in_headers[0].ptr<int32_t>(r, c);
            for(int i=1; i<=gauss_size/2; i++){
                parent_buf_center[i] = in_headers[0].ptr<int32_t>(r+i, c);
                parent_buf_center[-i] = in_headers[0].ptr<int32_t>(r-i, c);
            }
            for (; c < end_c; c++, buf_ptr++){
                int32_t local_sum = gauss_knl_uint32[gauss_size/2] * (*(parent_buf_center[0]));
                int gauss_knl_idx = gauss_size/2 + 1;
                for(int i=1; i<=gauss_size/2; i++, gauss_knl_idx++){
                    local_sum += gauss_knl_uint32[gauss_knl_idx] *
                            (*(parent_buf_center[i]) + *(parent_buf_center[-i]));
                }
                *buf_ptr = int16_t(local_sum >> (2*gauss_quant_bit));

                for(int i=0; i<gauss_size; i++){
                    parent_buf_ptr[i]++;
                }

                if(need_pyr_){
                    bool is_even_col = c % 2;
                    int c_down = c / 2 - padded_col / 2;
                    bool is_c_in_down_roi = (c_down >= 0 && c_down < out_headers[1].cols);
                    if(is_even_row && is_even_col && is_r_in_down_roi && is_c_in_down_roi){
                        out_headers[1].at<uint8_t>(r_down, c_down) = uint8_t(*buf_ptr);
                    }
                }

            }
        }
    }

    void update_simd(int start_r, int start_c, int end_r, int end_c) override {
        const int simd_step = mipp::N<int32_t>();
        const mipp::Reg<int32_t> zero32_v = int32_t(0);
        for(int r = start_r; r < end_r; r++){
            bool is_even_row = r % 2;
            int r_down = r / 2 - padded_row / 2;
            bool is_r_in_down_roi;
            if(need_pyr_){
                is_r_in_down_roi = (r_down >= 0 && r_down < out_headers[1].rows);
            }
            int c = start_c;
            int16_t *buf_ptr = out_headers[0].ptr<int16_t>(r - op_row/2, c - op_col/2);

            std::vector<int32_t*> parent_buf_ptr(gauss_size);
            int32_t** parent_buf_center = &parent_buf_ptr[gauss_size/2];
            parent_buf_center[0] = in_headers[0].ptr<int32_t>(r, c);
            for(int i=1; i<=gauss_size/2; i++){
                parent_buf_center[i] = in_headers[0].ptr<int32_t>(r+i, c);
                parent_buf_center[-i] = in_headers[0].ptr<int32_t>(r-i, c);
            }
            for (; c <= end_c - 2*simd_step; c += simd_step, buf_ptr += simd_step){
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

                mipp::Reg<int16_t> local_sum_int16 = mipp::pack<int32_t,int16_t>(local_sum, zero32_v);
                local_sum_int16.store(buf_ptr);

                for(int i=0; i<gauss_size; i++){
                    parent_buf_ptr[i] += simd_step;
                }

                if(need_pyr_){
                    int16_t *buf_ptr_local = buf_ptr;
                    for(int c_local = c; c_local < c + simd_step; c_local++, buf_ptr_local++){
                        bool is_even_col = c_local % 2;
                        int c_down = c_local / 2 - padded_col / 2;
                        bool is_c_in_down_roi = (c_down >= 0 && c_down < out_headers[1].cols);
                        if(is_even_row && is_even_col && is_r_in_down_roi && is_c_in_down_roi){
                            out_headers[1].at<uint8_t>(r_down, c_down) = uint8_t(*buf_ptr_local);
                        }
                    }
                }
            }
            for (; c < end_c; c++, buf_ptr++){
                int32_t local_sum = gauss_knl_uint32[gauss_size/2] * (*(parent_buf_center[0]));
                int gauss_knl_idx = gauss_size/2 + 1;
                for(int i=1; i<=gauss_size/2; i++, gauss_knl_idx++){
                    local_sum += gauss_knl_uint32[gauss_knl_idx] *
                            (*(parent_buf_center[i]) + *(parent_buf_center[-i]));
                }
                *buf_ptr = int16_t(local_sum >> (2*gauss_quant_bit));

                for(int i=0; i<gauss_size; i++){
                    parent_buf_ptr[i]++;
                }

                if(need_pyr_){
                    bool is_even_col = c % 2;
                    int c_down = c / 2 - padded_col / 2;
                    bool is_c_in_down_roi = (c_down >= 0 && c_down < out_headers[1].cols);
                    if(is_even_row && is_even_col && is_r_in_down_roi && is_c_in_down_roi){
                        out_headers[1].at<uint8_t>(r_down, c_down) = uint8_t(*buf_ptr);
                    }
                }
            }
        }
    }
    std::shared_ptr<FilterNode> clone() const override {
        std::shared_ptr<FilterNode> node_new = std::make_shared<Gauss5x1withPyrdownNode_32S_16S_4bit_smaller>(
                   down_img_, need_pyr_);
        node_new->padded_row = padded_row;
        node_new->padded_col = padded_col;
        node_new->which_buffer = which_buffer;
        return node_new;
    }

    void link_special_header(const cv::Rect &cur_roi) override {
        auto roi_down = cur_roi;
        roi_down.x /= 2;
        roi_down.y /= 2;
        roi_down.width /= 2;
        roi_down.height /= 2;

        assert(out_headers.size() == 1 && "sanity check");
        out_headers.push_back(down_img_(roi_down));
    }

    cv::Mat down_img_;
    void set_need_pyr(bool need){need_pyr_ = need;}
    bool need_pyr_ = true;
    const int gauss_quant_bit = 4; // should be larger if gauss_size is larger
    const int gauss_size = 5;
    int32_t gauss_knl_uint32[5];
};


class Sobel1x3SxxSyxNode_16S_16S : public FilterNode {
public:
    Sobel1x3SxxSyxNode_16S_16S(): FilterNode("sobel1x3_sxx_syx", CV_16S, 1, CV_16S, 2, 1, 3){}

    void update_simple(int start_r, int start_c, int end_r, int end_c) override {
        for(int r = start_r; r < end_r; r++){
            int c = start_c;
            int16_t *parent_buf_ptr = in_headers[0].ptr<int16_t>(r, c);
            int16_t *buf_ptr_0 = out_headers[0].ptr<int16_t>(r - op_row/2, c - op_col/2);
            int16_t *buf_ptr_1 = out_headers[1].ptr<int16_t>(r - op_row/2, c - op_col/2);
            for(; c < end_c; c++, parent_buf_ptr++, buf_ptr_0++, buf_ptr_1++){
                // sxx  -1 0 1
                *buf_ptr_0 = -*(parent_buf_ptr-1) + *(parent_buf_ptr+1);
                // syx   1 2 1
                *buf_ptr_1 = *(parent_buf_ptr-1) + 2*(*parent_buf_ptr) + *(parent_buf_ptr+1);
            }
        }
    }
    void update_simd(int start_r, int start_c, int end_r, int end_c) override {
        const int simd_step = mipp::N<int16_t>();
        const mipp::Reg<int16_t> two_int16 = int16_t(2);
        for(int r = start_r; r < end_r; r++){
            int c = start_c;
            int16_t *parent_buf_ptr = in_headers[0].ptr<int16_t>(r, c);
            int16_t *buf_ptr_0 = out_headers[0].ptr<int16_t>(r - op_row/2, c - op_col/2);
            int16_t *buf_ptr_1 = out_headers[1].ptr<int16_t>(r - op_row/2, c - op_col/2);
            for(; c <= end_c-simd_step; c+=simd_step, parent_buf_ptr+=simd_step,
                buf_ptr_0+=simd_step, buf_ptr_1+=simd_step){
                mipp::Reg<int16_t> p0(parent_buf_ptr-1);
                mipp::Reg<int16_t> p1(parent_buf_ptr);
                mipp::Reg<int16_t> p2(parent_buf_ptr+1);

                mipp::Reg<int16_t> sxx = p2 - p0;

                // bit shift is too dangorous for signed value
                // mipp::Reg<int16_t> syx = p2 + p0 + (p1 << 1);
                mipp::Reg<int16_t> syx = p2 + p0 + (p1 * two_int16);

                sxx.store(buf_ptr_0);
                syx.store(buf_ptr_1);
            }
            for(; c < end_c; c++, parent_buf_ptr++, buf_ptr_0++, buf_ptr_1++){
                // sxx  -1 0 1
                *buf_ptr_0 = -*(parent_buf_ptr-1) + *(parent_buf_ptr+1);
                // syx   1 2 1
                *buf_ptr_1 = *(parent_buf_ptr-1) + 2*(*parent_buf_ptr) + *(parent_buf_ptr+1);
            }
        }
    }
    std::shared_ptr<FilterNode> clone() const override {
        std::shared_ptr<FilterNode> node_new = std::make_shared<Sobel1x3SxxSyxNode_16S_16S>();
        node_new->padded_row = padded_row;
        node_new->padded_col = padded_col;
        node_new->which_buffer = which_buffer;
        return node_new;
    }
};

class Sobel3x1SxySyyNode_16S_16S : public FilterNode {
public:
    Sobel3x1SxySyyNode_16S_16S(): FilterNode("sobel3x1_sxy_syy", CV_16S, 2, CV_16S, 2, 3, 1){}
    void update_simple(int start_r, int start_c, int end_r, int end_c) override {
        for(int r = start_r; r < end_r; r++){
            int c = start_c;
            int16_t *parent_buf_ptr_0 = in_headers[0].ptr<int16_t>(r, c);
            int16_t *parent_buf_ptr_0_ = in_headers[0].ptr<int16_t>(r-1, c);
            int16_t *parent_buf_ptr_0__ = in_headers[0].ptr<int16_t>(r+1, c);

            int16_t *parent_buf_ptr_1 = in_headers[1].ptr<int16_t>(r, c);
            int16_t *parent_buf_ptr_1_ = in_headers[1].ptr<int16_t>(r-1, c);
            int16_t *parent_buf_ptr_1__ = in_headers[1].ptr<int16_t>(r+1, c);

            int16_t *buf_ptr_0 = out_headers[0].ptr<int16_t>(r - op_row/2, c - op_col/2);
            int16_t *buf_ptr_1 = out_headers[1].ptr<int16_t>(r - op_row/2, c - op_col/2);
            for(; c < end_c; c++, buf_ptr_0++, buf_ptr_1++, parent_buf_ptr_0++,
                parent_buf_ptr_0_++, parent_buf_ptr_0__++, parent_buf_ptr_1++,
                parent_buf_ptr_1_++, parent_buf_ptr_1__++){
                // sxy  1 2 1
                *buf_ptr_0 = *(parent_buf_ptr_0_) +
                        2*(*parent_buf_ptr_0) + *(parent_buf_ptr_0__);

                // syy  -1 0 1
                *buf_ptr_1 = -*(parent_buf_ptr_1_) + *(parent_buf_ptr_1__);
            }
        }
    }
    void update_simd(int start_r, int start_c, int end_r, int end_c) override {
        const int simd_step = mipp::N<int16_t>();
        const mipp::Reg<int16_t> two_int16 = int16_t(2);
        for(int r = start_r; r < end_r; r++){
            int c = start_c;
            int16_t *parent_buf_ptr_0 = in_headers[0].ptr<int16_t>(r, c);
            int16_t *parent_buf_ptr_0_ = in_headers[0].ptr<int16_t>(r-1, c);
            int16_t *parent_buf_ptr_0__ = in_headers[0].ptr<int16_t>(r+1, c);

            int16_t *parent_buf_ptr_1 = in_headers[1].ptr<int16_t>(r, c);
            int16_t *parent_buf_ptr_1_ = in_headers[1].ptr<int16_t>(r-1, c);
            int16_t *parent_buf_ptr_1__ = in_headers[1].ptr<int16_t>(r+1, c);

            int16_t *buf_ptr_0 = out_headers[0].ptr<int16_t>(r - op_row/2, c - op_col/2);
            int16_t *buf_ptr_1 = out_headers[1].ptr<int16_t>(r - op_row/2, c - op_col/2);
            for(; c <= end_c - simd_step; c+=simd_step, buf_ptr_0+=simd_step, buf_ptr_1+=simd_step,
                parent_buf_ptr_0+=simd_step, parent_buf_ptr_0_+=simd_step, parent_buf_ptr_0__+=simd_step,
                parent_buf_ptr_1+=simd_step, parent_buf_ptr_1_+=simd_step, parent_buf_ptr_1__+=simd_step){
                {
                    mipp::Reg<int16_t> p0(parent_buf_ptr_0_);
                    mipp::Reg<int16_t> p1(parent_buf_ptr_0);
                    mipp::Reg<int16_t> p2(parent_buf_ptr_0__);

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
            for(; c < end_c; c++, buf_ptr_0++, buf_ptr_1++, parent_buf_ptr_0++,
                parent_buf_ptr_0_++, parent_buf_ptr_0__++, parent_buf_ptr_1++,
                parent_buf_ptr_1_++, parent_buf_ptr_1__++){
                // sxy  1 2 1
                *buf_ptr_0 = *(parent_buf_ptr_0_) +
                        2*(*parent_buf_ptr_0) + *(parent_buf_ptr_0__);

                // syy  -1 0 1
                *buf_ptr_1 = -*(parent_buf_ptr_1_) + *(parent_buf_ptr_1__);
            }
        }
    }
    std::shared_ptr<FilterNode> clone() const override {
        std::shared_ptr<FilterNode> node_new = std::make_shared<Sobel3x1SxySyyNode_16S_16S>();
        node_new->padded_row = padded_row;
        node_new->padded_col = padded_col;
        node_new->which_buffer = which_buffer;
        return node_new;
    }
};

class Sobel3x1SxySyyNodeWithDxy_16S_16S : public FilterNode {
public:
    Sobel3x1SxySyyNodeWithDxy_16S_16S(cv::Mat dx, cv::Mat dy): 
        FilterNode("sobel3x1WithDxy_sxy_syy", CV_16S, 2, CV_16S, 2, 3, 1){
            dx_ = dx;
            dy_ = dy;
            have_special_headers = true;
        }
    void update_simple(int start_r, int start_c, int end_r, int end_c) override {
        for(int r = start_r; r < end_r; r++){
            int c = start_c;
            int16_t *parent_buf_ptr_0 = in_headers[0].ptr<int16_t>(r, c);
            int16_t *parent_buf_ptr_0_ = in_headers[0].ptr<int16_t>(r-1, c);
            int16_t *parent_buf_ptr_0__ = in_headers[0].ptr<int16_t>(r+1, c);

            int16_t *parent_buf_ptr_1 = in_headers[1].ptr<int16_t>(r, c);
            int16_t *parent_buf_ptr_1_ = in_headers[1].ptr<int16_t>(r-1, c);
            int16_t *parent_buf_ptr_1__ = in_headers[1].ptr<int16_t>(r+1, c);

            int16_t *buf_ptr_0 = out_headers[0].ptr<int16_t>(r - op_row/2, c - op_col/2);
            int16_t *buf_ptr_1 = out_headers[1].ptr<int16_t>(r - op_row/2, c - op_col/2);
            for(; c < end_c; c++, buf_ptr_0++, buf_ptr_1++, parent_buf_ptr_0++,
                parent_buf_ptr_0_++, parent_buf_ptr_0__++, parent_buf_ptr_1++,
                parent_buf_ptr_1_++, parent_buf_ptr_1__++){
                // sxy  1 2 1
                *buf_ptr_0 = *(parent_buf_ptr_0_) +
                        2*(*parent_buf_ptr_0) + *(parent_buf_ptr_0__);

                // syy  -1 0 1
                *buf_ptr_1 = -*(parent_buf_ptr_1_) + *(parent_buf_ptr_1__);
            }
        }
        cv::Rect dxy_roi(padded_col, padded_row, out_headers[2].cols, out_headers[2].rows);
        out_headers[0](dxy_roi).copyTo(out_headers[2]);
        out_headers[1](dxy_roi).copyTo(out_headers[3]);
    }
    void update_simd(int start_r, int start_c, int end_r, int end_c) override {
        const int simd_step = mipp::N<int16_t>();
        const mipp::Reg<int16_t> two_int16 = int16_t(2);
        for(int r = start_r; r < end_r; r++){
            int c = start_c;
            int16_t *parent_buf_ptr_0 = in_headers[0].ptr<int16_t>(r, c);
            int16_t *parent_buf_ptr_0_ = in_headers[0].ptr<int16_t>(r-1, c);
            int16_t *parent_buf_ptr_0__ = in_headers[0].ptr<int16_t>(r+1, c);

            int16_t *parent_buf_ptr_1 = in_headers[1].ptr<int16_t>(r, c);
            int16_t *parent_buf_ptr_1_ = in_headers[1].ptr<int16_t>(r-1, c);
            int16_t *parent_buf_ptr_1__ = in_headers[1].ptr<int16_t>(r+1, c);

            int16_t *buf_ptr_0 = out_headers[0].ptr<int16_t>(r - op_row/2, c - op_col/2);
            int16_t *buf_ptr_1 = out_headers[1].ptr<int16_t>(r - op_row/2, c - op_col/2);
            for(; c <= end_c - simd_step; c+=simd_step, buf_ptr_0+=simd_step, buf_ptr_1+=simd_step,
                parent_buf_ptr_0+=simd_step, parent_buf_ptr_0_+=simd_step, parent_buf_ptr_0__+=simd_step,
                parent_buf_ptr_1+=simd_step, parent_buf_ptr_1_+=simd_step, parent_buf_ptr_1__+=simd_step){
                {
                    mipp::Reg<int16_t> p0(parent_buf_ptr_0_);
                    mipp::Reg<int16_t> p1(parent_buf_ptr_0);
                    mipp::Reg<int16_t> p2(parent_buf_ptr_0__);

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
            for(; c < end_c; c++, buf_ptr_0++, buf_ptr_1++, parent_buf_ptr_0++,
                parent_buf_ptr_0_++, parent_buf_ptr_0__++, parent_buf_ptr_1++,
                parent_buf_ptr_1_++, parent_buf_ptr_1__++){
                // sxy  1 2 1
                *buf_ptr_0 = *(parent_buf_ptr_0_) +
                        2*(*parent_buf_ptr_0) + *(parent_buf_ptr_0__);

                // syy  -1 0 1
                *buf_ptr_1 = -*(parent_buf_ptr_1_) + *(parent_buf_ptr_1__);
            }
        }
        cv::Rect dxy_roi(padded_col, padded_row, out_headers[2].cols, out_headers[2].rows);
        out_headers[0](dxy_roi).copyTo(out_headers[2]);
        out_headers[1](dxy_roi).copyTo(out_headers[3]);
    }
    std::shared_ptr<FilterNode> clone() const override {
        std::shared_ptr<FilterNode> node_new = std::make_shared<Sobel3x1SxySyyNodeWithDxy_16S_16S>(dx_, dy_);
        node_new->padded_row = padded_row;
        node_new->padded_col = padded_col;
        node_new->which_buffer = which_buffer;
        return node_new;
    }
    void link_special_header(const cv::Rect &cur_roi) override {
        assert(out_headers.size() == 2 && "sanity check");
        out_headers.push_back(dx_(cur_roi));
        out_headers.push_back(dy_(cur_roi));
    }

    cv::Mat dx_, dy_;
};

class MagSqure1x1Node_16S_32S : public FilterNode {
public:
    MagSqure1x1Node_16S_32S() : FilterNode("mag_squre", CV_16S, 2, CV_32S, 1, 1, 1) {}
    void update_simple(int start_r, int start_c, int end_r, int end_c) override {
        for(int r = start_r; r < end_r; r++){
            int c = start_c;
            int16_t *parent_buf_ptr_0 = in_headers[0].ptr<int16_t>(r, c);
            int16_t *parent_buf_ptr_1 = in_headers[1].ptr<int16_t>(r, c);
            int32_t *buf_ptr = out_headers[0].ptr<int32_t>(r - op_row/2, c - op_col/2);
            for(; c < end_c; c++, parent_buf_ptr_0++, parent_buf_ptr_1++, buf_ptr++){
                int32_t dx = int32_t(*parent_buf_ptr_0);
                int32_t dy = int32_t(*parent_buf_ptr_1);
                *buf_ptr = dx*dx+dy*dy;
            }
        }
    }
    void update_simd(int start_r, int start_c, int end_r, int end_c) override {
        const int simd_step = mipp::N<int32_t>();
        for(int r = start_r; r < end_r; r++){
            int c = start_c;
            int16_t *parent_buf_ptr_0 = in_headers[0].ptr<int16_t>(r, c);
            int16_t *parent_buf_ptr_1 = in_headers[1].ptr<int16_t>(r, c);
            int32_t *buf_ptr = out_headers[0].ptr<int32_t>(r - op_row/2, c - op_col/2);

            for (; c <= end_c - 2*simd_step; c += simd_step, buf_ptr += simd_step, parent_buf_ptr_0 += simd_step,
                 parent_buf_ptr_1 += simd_step){
                mipp::Reg<int16_t> dx_int16(parent_buf_ptr_0);
                mipp::Reg<int16_t> dy_int16(parent_buf_ptr_1);

                mipp::Reg<int32_t> dx_int32 = mipp::cvt<int16_t, int32_t>(mipp::low<int16_t>(dx_int16.r));
                mipp::Reg<int32_t> dy_int32 = mipp::cvt<int16_t, int32_t>(mipp::low<int16_t>(dy_int16.r));

                mipp::Reg<int32_t> mag = dx_int32*dx_int32 + dy_int32*dy_int32;
                mag.store(buf_ptr);
            }
            for(; c < end_c; c++, parent_buf_ptr_0++, parent_buf_ptr_1++, buf_ptr++){
                int32_t dx = int32_t(*parent_buf_ptr_0);
                int32_t dy = int32_t(*parent_buf_ptr_1);
                *buf_ptr = dx*dx+dy*dy;
            }
        }
    }
    std::shared_ptr<FilterNode> clone() const override {
        std::shared_ptr<FilterNode> node_new = std::make_shared<MagSqure1x1Node_16S_32S>();
        node_new->padded_row = padded_row;
        node_new->padded_col = padded_col;
        node_new->which_buffer = which_buffer;
        return node_new;
    }
};

class MagPhaseQuant1x1Node_16S_8U : public FilterNode {
public:
    MagPhaseQuant1x1Node_16S_8U(int thresh = 60*60): mag_thresh_l2(thresh),
        FilterNode("mag_phase_quant1x1", CV_16S, 2, CV_8U, 1, 1, 1){}
    void update_simple(int start_r, int start_c, int end_r, int end_c) override {
        for(int r = start_r; r < end_r; r++){
            int c = start_c;
            int16_t *parent_buf_ptr_0 = in_headers[0].ptr<int16_t>(r, c);
            int16_t *parent_buf_ptr_1 = in_headers[1].ptr<int16_t>(r, c);
            uint8_t *buf_ptr = out_headers[0].ptr<uint8_t>(r, c);
            for(; c < end_c; c++, parent_buf_ptr_0++, parent_buf_ptr_1++, buf_ptr++){
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
                    *buf_ptr = label;
                }else{
                    *buf_ptr = INVALID;
                }
            }
        }
    }

    void update_simd(int start_r, int start_c, int end_r, int end_c) override {
        const int simd_step = mipp::N<int32_t>();
        const mipp::Reg<int16_t> ZERO16_v = int16_t(0);
        const mipp::Reg<int32_t> ZERO32_v = int32_t(0);
        const mipp::Reg<int32_t> ONE32_v = int32_t(1);
        const mipp::Reg<int32_t> TWO32_v = int32_t(2);
        const mipp::Reg<int32_t> THREE32_v = int32_t(3);
        const mipp::Reg<int32_t> FOUR32_v = int32_t(4);
        const mipp::Reg<int32_t> EIGHT32_v = int32_t(8);
        const mipp::Reg<int32_t> TG1125_v = TG1125;
        const mipp::Reg<int32_t> TG3375_v = TG3375;
        const mipp::Reg<int32_t> TG5625_v = TG5625;
        const mipp::Reg<int32_t> TG7875_v = TG7875;
        const mipp::Reg<int32_t> INVALID_v = int32_t(INVALID);
        for(int r = start_r; r < end_r; r++){
            int c = start_c;
            int16_t *parent_buf_ptr_0 = in_headers[0].ptr<int16_t>(r, c);
            int16_t *parent_buf_ptr_1 = in_headers[1].ptr<int16_t>(r, c);
            uint8_t *buf_ptr = out_headers[0].ptr<uint8_t>(r, c);
            for(; c <= end_c - 4*simd_step; c+=simd_step, parent_buf_ptr_0+=simd_step,
                parent_buf_ptr_1+=simd_step, buf_ptr+=simd_step){
                mipp::Reg<int16_t> dx_int16(parent_buf_ptr_0);
                mipp::Reg<int16_t> dy_int16(parent_buf_ptr_1);

                mipp::Reg<int32_t> dx_int32 = mipp::cvt<int16_t, int32_t>(mipp::low<int16_t>(dx_int16.r));
                mipp::Reg<int32_t> dy_int32 = mipp::cvt<int16_t, int32_t>(mipp::low<int16_t>(dy_int16.r));

                mipp::Reg<int32_t> mag = dx_int32*dx_int32 + dy_int32*dy_int32;

                auto mag_mask = mag > mag_thresh_l2;
                if(!mipp::testz(mag_mask)){
                    mipp::Reg<int32_t> abs_dx = mipp::abs(dx_int32);
                    mipp::Reg<int32_t> abs_dy = mipp::abs(dy_int32) << 15;

//                    uint8_t label =  (y<x*TG3375)?
//                                    ((y<x*TG1125)?(0):(1)):
//                                    ((y<x*TG5625)?(2):
//                                    ((y<x*TG7875)?(3):(4)));
                    mipp::Reg<int32_t> label_v =
                                mipp::blend(
                                mipp::blend(ZERO32_v, ONE32_v, abs_dy<abs_dx*TG1125_v),
                                mipp::blend(TWO32_v,
                                mipp::blend(THREE32_v, FOUR32_v, abs_dy<abs_dx*TG7875_v)
                                            , abs_dy<abs_dx*TG5625_v), abs_dy<abs_dx*TG3375_v);
                    label_v = mipp::blend(label_v, EIGHT32_v-label_v,
                                         ((label_v == ZERO32_v) |
                                         (((dx_int32>0) & (dy_int32>0))|((dx_int32<0) & (dy_int32<0)))));

                    label_v = mipp::blend(label_v, INVALID_v, mag_mask);

                    // range 0-7, so no worry for signed int while pack
                    mipp::Reg<int16_t> label16_v = mipp::pack<int32_t, int16_t>(label_v, ZERO32_v);
                    mipp::Reg<int8_t> label8_v = mipp::pack<int16_t, int8_t>(label16_v, ZERO16_v);

                    label8_v.store((int8_t*)buf_ptr);
                }else{
                    mipp::Reg<int8_t> label8_v = int8_t(INVALID);
                    label8_v.store((int8_t*)buf_ptr);
                }
            }
            for(; c < end_c; c++, parent_buf_ptr_0++, parent_buf_ptr_1++, buf_ptr++){
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
                    *buf_ptr = label;
                }else{
                    *buf_ptr = INVALID;
                }
            }
        }
    }

    void set_mag_thresh(int thresh){mag_thresh_l2 = thresh;}
    std::shared_ptr<FilterNode> clone() const override {
        std::shared_ptr<FilterNode> node_new = std::make_shared<MagPhaseQuant1x1Node_16S_8U>(mag_thresh_l2);
        node_new->padded_row = padded_row;
        node_new->padded_col = padded_col;
        node_new->which_buffer = which_buffer;
        return node_new;
    }
    int mag_thresh_l2;
    const int32_t TG1125 = std::round(std::tan(11.25/180*CV_PI)*(1<<15));
    const int32_t TG3375 = std::round(std::tan(33.75/180*CV_PI)*(1<<15));
    const int32_t TG5625 = std::round(std::tan(56.25/180*CV_PI)*(1<<15));
    const int32_t TG7875 = std::round(std::tan(78.75/180*CV_PI)*(1<<15));
};

class MagPhaseQuantShift1x1Node_16S_8U : public FilterNode {
public:
    MagPhaseQuantShift1x1Node_16S_8U(int thresh = 60*60): mag_thresh_l2(thresh),
        FilterNode("mag_phase_quant1x1", CV_16S, 2, CV_8U, 1, 1, 1){}
    void update_simple(int start_r, int start_c, int end_r, int end_c) override {
        for(int r = start_r; r < end_r; r++){
            int c = start_c;
            int16_t *parent_buf_ptr_0 = in_headers[0].ptr<int16_t>(r, c);
            int16_t *parent_buf_ptr_1 = in_headers[1].ptr<int16_t>(r, c);
            int16_t *buf_ptr = out_headers[0].ptr<int16_t>(r, c);
            for(; c < end_c; c++, parent_buf_ptr_0++, parent_buf_ptr_1++, buf_ptr++){
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
                    *buf_ptr = uint8_t(uint8_t(1)<<label);
                }else{
                    *buf_ptr = 0;
                }
            }
        }
    }
    void update_simd(int start_r, int start_c, int end_r, int end_c) override {
        const int simd_step = mipp::N<int32_t>();
        const mipp::Reg<int16_t> ZERO16_v = int16_t(0);
        const mipp::Reg<int32_t> ZERO32_v = int32_t(0);
        const mipp::Reg<int32_t> ONE32_v = int32_t(1);
        const mipp::Reg<int32_t> TWO32_v = int32_t(2);
        const mipp::Reg<int32_t> THREE32_v = int32_t(3);
        const mipp::Reg<int32_t> FOUR32_v = int32_t(4);
        const mipp::Reg<int32_t> EIGHT32_v = int32_t(8);
        const mipp::Reg<int32_t> TG1125_v = TG1125;
        const mipp::Reg<int32_t> TG3375_v = TG3375;
        const mipp::Reg<int32_t> TG5625_v = TG5625;
        const mipp::Reg<int32_t> TG7875_v = TG7875;
        const mipp::Reg<int32_t> INVALID_v = int32_t(INVALID);
        for(int r = start_r; r < end_r; r++){
            int c = start_c;
            int16_t *parent_buf_ptr_0 = in_headers[0].ptr<int16_t>(r, c);
            int16_t *parent_buf_ptr_1 = in_headers[1].ptr<int16_t>(r, c);
            uint8_t *buf_ptr = out_headers[0].ptr<uint8_t>(r, c);
            for(; c <= end_c - 4*simd_step; c+=simd_step, parent_buf_ptr_0+=simd_step,
                parent_buf_ptr_1+=simd_step, buf_ptr+=simd_step){
                mipp::Reg<int16_t> dx_int16(parent_buf_ptr_0);
                mipp::Reg<int16_t> dy_int16(parent_buf_ptr_1);

                mipp::Reg<int32_t> dx_int32 = mipp::cvt<int16_t, int32_t>(mipp::low<int16_t>(dx_int16.r));
                mipp::Reg<int32_t> dy_int32 = mipp::cvt<int16_t, int32_t>(mipp::low<int16_t>(dy_int16.r));

                mipp::Reg<int32_t> mag = dx_int32*dx_int32 + dy_int32*dy_int32;

                auto mag_mask = mag > mag_thresh_l2;
                if(!mipp::testz(mag_mask)){
                    mipp::Reg<int32_t> abs_dx = mipp::abs(dx_int32);
                    mipp::Reg<int32_t> abs_dy = mipp::abs(dy_int32) << 15;

//                    uint8_t label =  (y<x*TG3375)?
//                                    ((y<x*TG1125)?(0):(1)):
//                                    ((y<x*TG5625)?(2):
//                                    ((y<x*TG7875)?(3):(4)));
                    mipp::Reg<int32_t> label_v =
                                mipp::blend(
                                mipp::blend(ZERO32_v, ONE32_v, abs_dy<abs_dx*TG1125_v),
                                mipp::blend(TWO32_v,
                                mipp::blend(THREE32_v, FOUR32_v, abs_dy<abs_dx*TG7875_v)
                                            , abs_dy<abs_dx*TG5625_v), abs_dy<abs_dx*TG3375_v);
                    label_v = mipp::blend(label_v, EIGHT32_v-label_v,
                                         ((label_v == ZERO32_v) |
                                         (((dx_int32>0) & (dy_int32>0))|((dx_int32<0) & (dy_int32<0)))));

                    label_v = mipp::blend(label_v, INVALID_v, mag_mask);

                    // range 0-7, so no worry for signed int while pack
                    mipp::Reg<int16_t> label16_v = mipp::pack<int32_t, int16_t>(label_v, ZERO32_v);
                    mipp::Reg<int8_t> label8_v = mipp::pack<int16_t, int8_t>(label16_v, ZERO16_v);

                    uint8_t temp_result[mipp::N<int8_t>()] = {0};
                    label8_v.store((int8_t*)temp_result);
                    for(int j=0; j<simd_step; j++){
                        if(temp_result[j] != INVALID)
                            buf_ptr[j] = uint8_t(uint8_t(1)<<temp_result[j]);
                        else
                            buf_ptr[j] = 0;
                    }
                }else{
                    mipp::Reg<int8_t> label8_v = int8_t(0);
                    label8_v.store((int8_t*)buf_ptr);
                }
            }
            for(; c < end_c; c++, parent_buf_ptr_0++, parent_buf_ptr_1++, buf_ptr++){
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
                    *buf_ptr = uint8_t(uint8_t(1)<<label);
                }else{
                    *buf_ptr = 0;
                }
            }
        }
    }
    void set_mag_thresh(int thresh){mag_thresh_l2 = thresh;}
    std::shared_ptr<FilterNode> clone() const override {
        std::shared_ptr<FilterNode> node_new = std::make_shared<MagPhaseQuantShift1x1Node_16S_8U>(mag_thresh_l2);
        node_new->padded_row = padded_row;
        node_new->padded_col = padded_col;
        node_new->which_buffer = which_buffer;
        return node_new;
    }
    int mag_thresh_l2;
    const int32_t TG1125 = std::round(std::tan(11.25/180*CV_PI)*(1<<15));
    const int32_t TG3375 = std::round(std::tan(33.75/180*CV_PI)*(1<<15));
    const int32_t TG5625 = std::round(std::tan(56.25/180*CV_PI)*(1<<15));
    const int32_t TG7875 = std::round(std::tan(78.75/180*CV_PI)*(1<<15));
};

class Hist3x3Node_8U_8U : public FilterNode {
public:
    Hist3x3Node_8U_8U() : FilterNode("hist3x3", CV_8U, 1, CV_8U, 1, 3, 3){}
    void update_simple(int start_r, int start_c, int end_r, int end_c) override {

        for(int r = start_r; r < end_r; r++){
            int c = start_c;
            uint8_t *parent_buf_ptr = in_headers[0].ptr<uint8_t>(r, c);
            uint8_t *parent_buf_ptr_ = in_headers[0].ptr<uint8_t>(r-1, c);
            uint8_t *parent_buf_ptr__ = in_headers[0].ptr<uint8_t>(r+1, c);

            uint8_t *buf_ptr = out_headers[0].ptr<uint8_t>(r - op_row/2, c - op_col/2);
            for(; c < end_c; c++, buf_ptr++, parent_buf_ptr++, parent_buf_ptr_++, parent_buf_ptr__++){

                uint8_t votes_of_ori[8] = {0};
                if(*parent_buf_ptr != INVALID) votes_of_ori[*parent_buf_ptr]++;
                if(*(parent_buf_ptr+1) != INVALID) votes_of_ori[*(parent_buf_ptr+1)]++;
                if(*(parent_buf_ptr-1) != INVALID) votes_of_ori[*(parent_buf_ptr-1)]++;
                if(*(parent_buf_ptr_) != INVALID) votes_of_ori[*(parent_buf_ptr_)]++;
                if(*(parent_buf_ptr_+1) != INVALID) votes_of_ori[*(parent_buf_ptr_+1)]++;
                if(*(parent_buf_ptr_-1) != INVALID) votes_of_ori[*(parent_buf_ptr_-1)]++;
                if(*(parent_buf_ptr__) != INVALID) votes_of_ori[*(parent_buf_ptr__)]++;
                if(*(parent_buf_ptr__+1) != INVALID) votes_of_ori[*(parent_buf_ptr__+1)]++;
                if(*(parent_buf_ptr__-1) != INVALID) votes_of_ori[*(parent_buf_ptr__-1)]++;

                // Find bin with the most votes from the patch
                int max_votes = 0;
                uint8_t index = 0;
                for (uint8_t i = 0; i < 8; ++i){
                    if (max_votes < votes_of_ori[i]){
                        index = i;
                        max_votes = votes_of_ori[i];
                    }
                }

                // Only accept the quantization if majority of pixels in the patch agree
                if (max_votes >= NEIGHBOR_THRESHOLD) *buf_ptr = uint8_t(uint8_t(1) << index);
                else *buf_ptr = 0;
            }
        }
    }
    void update_simd(int start_r, int start_c, int end_r, int end_c) override {
        update_simple(start_r, start_c, end_r, end_c);
    }
    std::shared_ptr<FilterNode> clone() const override {
        std::shared_ptr<FilterNode> node_new = std::make_shared<Hist3x3Node_8U_8U>();
        node_new->padded_row = padded_row;
        node_new->padded_col = padded_col;
        node_new->which_buffer = which_buffer;
        return node_new;
    }
    const int NEIGHBOR_THRESHOLD = 5;
};

class Spread1xnNode_8U_8U : public FilterNode {
public:
    Spread1xnNode_8U_8U(int n_temp) : FilterNode("spread1xn", CV_8U, 1, CV_8U, 1, 1, n_temp), n_spread(n_temp) {}
    void update_simple(int start_r, int start_c, int end_r, int end_c) override {
        for(int r = start_r; r < end_r; r++){
            int c = start_c;
            uint8_t *parent_buf_ptr = in_headers[0].ptr<uint8_t>(r, c);
            uint8_t *buf_ptr = out_headers[0].ptr<uint8_t>(r - op_row/2, c - op_col/2);
            for(; c < end_c; c++, parent_buf_ptr++, buf_ptr++){
                uint8_t local_sum = *(parent_buf_ptr);
                for(int i=1; i<=op_col/2; i++){
                    local_sum |= *(parent_buf_ptr+i);
                    local_sum |= *(parent_buf_ptr-i);
                }
                *buf_ptr = local_sum;
            }
        }
    }
    void update_simd(int start_r, int start_c, int end_r, int end_c) override{
        const int simd_step = mipp::N<int8_t>();
        for(int r = start_r; r < end_r; r++){
            int c = start_c;
            uint8_t *parent_buf_ptr = in_headers[0].ptr<uint8_t>(r, c);
            uint8_t *buf_ptr = out_headers[0].ptr<uint8_t>(r - op_row/2, c - op_col/2);
            for(; c <= end_c-simd_step; c+=simd_step, parent_buf_ptr+=simd_step, buf_ptr+=simd_step){
                mipp::Reg<uint8_t> local_sum(parent_buf_ptr);
                for(int i=1; i<=op_col/2; i++){
                    mipp::Reg<uint8_t> src_v0(parent_buf_ptr+i);
                    local_sum = mipp::orb(src_v0, local_sum);

                    mipp::Reg<uint8_t> src_v1(parent_buf_ptr-i);
                    local_sum = mipp::orb(src_v1, local_sum);
                }
                local_sum.store(buf_ptr);
            }
            for(; c < end_c; c++, parent_buf_ptr++, buf_ptr++){
                uint8_t local_sum = *(parent_buf_ptr);
                for(int i=1; i<=op_col/2; i++){
                    local_sum |= *(parent_buf_ptr+i);
                    local_sum |= *(parent_buf_ptr-i);
                }
                *buf_ptr = local_sum;
            }
        }
    }
    std::shared_ptr<FilterNode> clone() const override {
        std::shared_ptr<FilterNode> node_new = std::make_shared<Spread1xnNode_8U_8U>(n_spread);
        node_new->padded_row = padded_row;
        node_new->padded_col = padded_col;
        node_new->which_buffer = which_buffer;
        return node_new;
    }
    int n_spread;
};

class Spreadnx1Node_8U_8U : public FilterNode {
public:
    Spreadnx1Node_8U_8U(int n_temp) : FilterNode("spreadnx1", CV_8U, 1, CV_8U, 1, n_temp, 1), n_spread(n_temp) {}
    void update_simple(int start_r, int start_c, int end_r, int end_c) override {
        for(int r = start_r; r < end_r; r++){
            int c = start_c;

            std::vector<uint8_t*> parent_buf_ptr(op_row);
            uint8_t** parent_buf_ptr_center = &parent_buf_ptr[op_row/2];
            parent_buf_ptr_center[0] = in_headers[0].ptr<uint8_t>(r, c);
            for(int i=1; i<=op_row/2; i++){
                parent_buf_ptr_center[+i] = in_headers[0].ptr<uint8_t>(r+i, c);
                parent_buf_ptr_center[-i] = in_headers[0].ptr<uint8_t>(r-i, c);
            }
            uint8_t *buf_ptr = out_headers[0].ptr<uint8_t>(r - op_row/2, c - op_col/2);
            for (; c < end_c; c++, buf_ptr++){
                uint8_t local_sum =  *(parent_buf_ptr_center[0]);
                for(int i=1; i<=op_row/2; i++){
                    local_sum |= *(parent_buf_ptr_center[i]);
                    local_sum |= *(parent_buf_ptr_center[-i]);
                }
                *buf_ptr = local_sum;

                for(int i=0; i<op_row; i++)
                    parent_buf_ptr[i]++;
            }
        }
    }
    void update_simd(int start_r, int start_c, int end_r, int end_c) override{
        const int simd_step = mipp::N<int8_t>();
        for(int r = start_r; r < end_r; r++){
            int c = start_c;
            std::vector<uint8_t*> parent_buf_ptr(op_row);
            uint8_t** parent_buf_ptr_center = &parent_buf_ptr[op_row/2];
            parent_buf_ptr_center[0] = in_headers[0].ptr<uint8_t>(r, c);
            for(int i=1; i<=op_row/2; i++){
                parent_buf_ptr_center[+i] = in_headers[0].ptr<uint8_t>(r+i, c);
                parent_buf_ptr_center[-i] = in_headers[0].ptr<uint8_t>(r-i, c);
            }
            uint8_t *buf_ptr = out_headers[0].ptr<uint8_t>(r - op_row/2, c - op_col/2);
            for (; c <= end_c-simd_step; c += simd_step, buf_ptr += simd_step){
                mipp::Reg<uint8_t> local_sum(parent_buf_ptr_center[0]);
                for(int i=1; i<=op_row/2; i++){
                    mipp::Reg<uint8_t> src_v0(parent_buf_ptr_center[i]);
                    local_sum = mipp::orb(src_v0, local_sum);

                    mipp::Reg<uint8_t> src_v1(parent_buf_ptr_center[-i]);
                    local_sum = mipp::orb(src_v1, local_sum);
                }
                local_sum.store(buf_ptr);
                for(int i=0; i<op_row; i++)
                    parent_buf_ptr[i]+=simd_step;
            }
            for (; c < end_c; c++, buf_ptr++){
                uint8_t local_sum =  *(parent_buf_ptr_center[0]);
                for(int i=1; i<=op_row/2; i++){
                    local_sum |= *(parent_buf_ptr_center[i]);
                    local_sum |= *(parent_buf_ptr_center[-i]);
                }
                *buf_ptr = local_sum;

                for(int i=0; i<op_row; i++)
                    parent_buf_ptr[i]++;
            }
        }
    }
    std::shared_ptr<FilterNode> clone() const override {
        std::shared_ptr<FilterNode> node_new = std::make_shared<Spreadnx1Node_8U_8U>(n_spread);
        node_new->padded_row = padded_row;
        node_new->padded_col = padded_col;
        node_new->which_buffer = which_buffer;
        return node_new;
    }
    int n_spread;
};

class Response1x1Node_8U_8U : public FilterNode {
public:
    Response1x1Node_8U_8U() : FilterNode("response1x1", CV_8U, 1, CV_8U, 8, 1, 1) {}

    void update_simple(int start_r, int start_c, int end_r, int end_c) override {
        for(int ori=0; ori<8; ori++) {
            for(int r = start_r; r < end_r; r++){
                int c = start_c;
                uint8_t *parent_buf_ptr = in_headers[0].ptr<uint8_t>(r, c);
                uint8_t *buf_ptr = out_headers[ori].ptr<uint8_t>(r, c);
                for(; c < end_c; c++, parent_buf_ptr++, buf_ptr++){
                    *buf_ptr = (hit_mask[ori] & *parent_buf_ptr) ? scores[0] :
                            ((side_mask[ori] & *parent_buf_ptr) ? scores[1] : 0);
                }
            }
        }
    }
    void update_simd(int start_r, int start_c, int end_r, int end_c) override {
        const int simd_step = mipp::N<int8_t>();
        const mipp::Reg<uint8_t> const_score0 = scores[0];
        const mipp::Reg<uint8_t> const_score1 = scores[1];
        const mipp::Reg<uint8_t> zero8_v = uint8_t(0);
        for(int ori=0; ori<8; ori++) {
            const mipp::Reg<uint8_t> hit_mask_v = hit_mask[ori];
            const mipp::Reg<uint8_t> side_mask_v = side_mask[ori];
            for(int r = start_r; r < end_r; r++){
                int c = start_c;
                uint8_t *parent_buf_ptr = in_headers[0].ptr<uint8_t>(r, c);
                uint8_t *buf_ptr = out_headers[ori].ptr<uint8_t>(r, c);
                for(; c <= end_c-simd_step; c+=simd_step, parent_buf_ptr+=simd_step, buf_ptr+=simd_step){
                    mipp::Reg<uint8_t> src_v(parent_buf_ptr);

                    auto result = mipp::blend(mipp::blend(zero8_v, const_score1, (side_mask_v & src_v) == zero8_v),
                                              const_score0, (hit_mask_v & src_v) == zero8_v);
                    result.store(buf_ptr);
                }
                for(; c < end_c; c++, parent_buf_ptr++, buf_ptr++){
                    *buf_ptr = (hit_mask[ori] & *parent_buf_ptr) ? scores[0] :
                            ((side_mask[ori] & *parent_buf_ptr) ? scores[1] : 0);
                }
            }
        }
    }
    std::shared_ptr<FilterNode> clone() const override {
        std::shared_ptr<FilterNode> node_new = std::make_shared<Response1x1Node_8U_8U>();
        node_new->padded_row = padded_row;
        node_new->padded_col = padded_col;
        node_new->which_buffer = which_buffer;
        return node_new;
    }
    const uint8_t scores[2] = {4, 3};
    const uint8_t hit_mask[8] = { 1,   2, 4,  8,  16, 32, 64,  128};
    const uint8_t side_mask[8] = {130, 5, 10, 20, 40, 80, 160, 65};
};

class LinearizeTxTNode_8U_8U : public FilterNode {
public:
    LinearizeTxTNode_8U_8U(int cur_T_temp, int imgCols_temp, std::vector<cv::Mat> buffer) : buffer_(buffer),
        FilterNode("linearizeTxT", CV_8U, 8, CV_8U, 8, 1, 1), cur_T(cur_T_temp), imgCols(imgCols_temp){
        linearize_row_step = imgCols / cur_T;
        have_special_headers = true;
    }

    void link_special_header(const cv::Rect &cur_roi) override {
        out_headers.clear();
        for(int ori=0; ori<8; ori++){
            out_headers.push_back(buffer_[ori]);
        }
    }

    void update_simple(int start_r, int start_c, int end_r, int end_c) override {
        for(int ori = 0; ori < 8; ori++){
            int global_start_r = start_r + cur_row;
            int global_start_c = start_c + cur_col;

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
            // int target_c = linearize_row_step * (r / cur_T) + c / cur_T;
            // int target_r = cur_T * (r % cur_T) + c % cur_T;

            // cleaner codes, but heavy math operation in the innermost loop
//            int c = start_c;
//            for (int r = start_r; r < end_r; r++){
//                c = start_c;
//                uint8_t *parent_buf_ptr = out_headers[ori].ptr<uint8_t>(r, c, ori);
//                for (; c < end_c; c++, parent_buf_ptr++){
//                    int target_c = linearize_row_step * (r / cur_T) + c / cur_T;
//                    int target_r = cur_T * (r % cur_T) + c % cur_T;
//                    in_headers[ori].at<uint8_t>(target_r, target_c) = *parent_buf_ptr;
//                }
//            }

            // more codes, but less operation int the innermost loop
            assert(start_c % cur_T == 0);
            assert(cur_col % cur_T == 0);
            int target_start_c = linearize_row_step * (global_start_r / cur_T) + global_start_c / cur_T;
            int target_start_r = cur_T * (global_start_r % cur_T) + global_start_c % cur_T;

            for(int tileT_r = start_r; tileT_r < start_r + cur_T; ++tileT_r){
                for (int tileT_c = start_c; tileT_c < start_c + cur_T; ++tileT_c){
                    uint8_t *memory = out_headers[ori].ptr<uint8_t>(target_start_r, target_start_c);
                    target_start_r++;
                    if(target_start_r >= cur_T * cur_T){
                        target_start_r -= cur_T * cur_T;
                        target_start_c += linearize_row_step;
                    }

                    // Inner two loops copy every T-th pixel into the linear memory
                    for(int r = tileT_r; r < end_r; r += cur_T, memory += linearize_row_step){
                        uint8_t *parent_buf_ptr = in_headers[ori].ptr<uint8_t>(r);
                        uint8_t *local_memory = memory;
                        for (int c = tileT_c; c < end_c; c += cur_T, local_memory++)
                            *local_memory = parent_buf_ptr[c];
                    }
                }
            }
        }
    }

    void update_simd(int start_r, int start_c, int end_r, int end_c) override {
        update_simple(start_r, start_c, end_r, end_c);
    }
    std::shared_ptr<FilterNode> clone() const override {
        std::shared_ptr<FilterNode> node_new = std::make_shared<LinearizeTxTNode_8U_8U>(cur_T, imgCols, buffer_);
        node_new->padded_row = padded_row;
        node_new->padded_col = padded_col;
        node_new->which_buffer = which_buffer;
        return node_new;
    }
    int cur_T;
    int imgCols;
    int linearize_row_step;
    std::vector<cv::Mat> buffer_;
};

class ProcessManager {
public:
    ProcessManager(int tileRows = 32, int tileCols = 256): tileRows_(tileRows), tileCols_(tileCols) {}
    void set_num_threads(int t){num_threads_ = t;}
    std::vector<std::shared_ptr<FilterNode>>& get_nodes(){return nodes_;}

    void arrange(int outRows, int outCols){
        if(nodes_.empty()){
            std::cout << "no nodes yet" << std::endl;
            return;
        }

        // ping-pong buffer
        for(int i=0; i<nodes_.size(); i++){
            nodes_[i]->which_buffer = i % 2;
        }

        { // tile ROIs
            update_rois_.clear();
            int cur_row = 0;
            for(; cur_row <= outRows - tileRows_; cur_row += tileRows_){
                int cur_col = 0;
                for(; cur_col <= outCols - tileCols_; cur_col += tileCols_){
                    update_rois_.push_back({cur_col, cur_row, tileCols_, tileRows_});
                }
                if(cur_col < outCols){
                    update_rois_.push_back({cur_col, cur_row, outCols - cur_col, tileRows_});
                }
            }
            if(cur_row < outRows){
                int cur_col = 0;
                for(; cur_col <= outCols - tileCols_; cur_col += tileCols_){
                    update_rois_.push_back({cur_col, cur_row, tileCols_, outRows - cur_row});
                }
                if(cur_col < outCols){
                    update_rois_.push_back({cur_col, cur_row, outCols - cur_col, outRows - cur_row});
                }
            }
        }
        { // calculate paddings: IN(tileBuffer0) --Filter0-- tileBuffer1 --Filter1-- tileBuffer0 ...
            nodes_.back()->padded_row = nodes_.back()->op_row/2;
            nodes_.back()->padded_col = nodes_.back()->op_col/2;
            for(int i=nodes_.size()-2; i>=0; i--){
                nodes_[i]->padded_row = nodes_[i+1]->padded_row + nodes_[i]->op_row/2;
                nodes_[i]->padded_col = nodes_[i+1]->padded_col + nodes_[i]->op_col/2;
            }
        }
        { // calculate max memory footprint
            maxMemoFootprint_ = 0;
            for(auto& node: nodes_){
                int tile_size = (tileRows_ + node->padded_row * 2) * (tileCols_ + node->padded_col * 2);
                int type_size = CvTypeSize(node->input_type);
                int footprint = tile_size * type_size * node->input_num + node->input_num * 128; // with some alignments
                if(footprint > maxMemoFootprint_) maxMemoFootprint_ = footprint;
            }
        }
    }

    template<typename T>
    void copyToWith0Bound(const cv::Mat& in, cv::Mat& out, const cv::Rect& roi, int rr_ori, int cc_ori){
        for(int r=roi.y, rr=rr_ori; r<roi.y + roi.height; r++, rr++){
            T* out_ptr = out.ptr<T>(rr);
            for(int c=roi.x, cc=cc_ori; c<roi.x + roi.width; c++, cc++){
                if(r<0 || r>=in.rows || c<0 || c>=in.cols){
                    out_ptr[cc] = 0;
                    continue;
                }else{
                    const T* in_ptr = in.ptr<T>(r);
                    out_ptr[cc] = in_ptr[c];
                }
            }
        }
    }

    void process(const std::vector<cv::Mat>& in_v, const std::vector<cv::Mat>& out_v){

    // sanity check
    assert(!in_v.empty() && "empty input");
    assert(!out_v.empty() && "empty output");

    assert(!in_v[0].empty() && "input memory should be held by user");
    assert(!out_v[0].empty() && "output memory should be held by user");

//    if(!check_if_nodes_valid()){
//        std::cout << "nodes are not compatible !!!" << std::endl;
//        return;
//    }
    assert(nodes_[0]->input_num == in_v.size() &&
            nodes_[0]->input_type == in_v[0].type() && "first node is not compatible");
    assert(nodes_.back()->output_num == out_v.size() &&
           nodes_.back()->output_type == out_v[0].type() && "last node is not compatible");

#ifdef _OPENMP
#pragma omp parallel num_threads(num_threads_)
    {
#endif
    // prepare private buffer here
    std::vector<char*> filter_buffer(2);
    char* buffer_0 = new char[maxMemoFootprint_];
    char* buffer_1 = new char[maxMemoFootprint_];
    filter_buffer[0] = buffer_0 + aligned256_after_n_char(buffer_0);
    filter_buffer[1] = buffer_1 + aligned256_after_n_char(buffer_1);

    auto nodes_private = deep_copy_shared_ptr_vec(nodes_);

#ifdef _OPENMP
#pragma omp for nowait
#endif
    for (int roi_iter = 0; roi_iter < update_rois_.size(); ++roi_iter)
    {
        auto cur_roi = update_rois_[roi_iter];
        { // copy in_v to buffer
            char* buffer_cur = filter_buffer[0];
            nodes_private[0]->in_headers.clear();
            for(auto& in: in_v){
                // data is not owned by mat if initialize it this way
                cv::Mat buffer_header(cur_roi.height + nodes_private[0]->padded_row * 2,
                        cur_roi.width + nodes_private[0]->padded_col * 2, in.type(), buffer_cur);

                auto cur_roi_padded = cur_roi;
                cur_roi_padded.x -= nodes_private[0]->padded_col;
                cur_roi_padded.y -= nodes_private[0]->padded_row;
                cur_roi_padded.width += 2*nodes_private[0]->padded_col;
                cur_roi_padded.height += 2*nodes_private[0]->padded_row;

                if(in.type() == CV_8U) copyToWith0Bound<uchar>(in, buffer_header, cur_roi_padded, 0, 0);
                else if(in.type() == CV_16S) copyToWith0Bound<int16_t>(in, buffer_header, cur_roi_padded, 0, 0);
                else if(in.type() == CV_16U) copyToWith0Bound<uint16_t>(in, buffer_header, cur_roi_padded, 0, 0);
                else if(in.type() == CV_32S) copyToWith0Bound<int32_t>(in, buffer_header, cur_roi_padded, 0, 0);
                else if(in.type() == CV_32F) copyToWith0Bound<float>(in, buffer_header, cur_roi_padded, 0, 0);
                else if(in.type() == CV_8UC3) copyToWith0Bound<cv::Vec3b>(in, buffer_header, cur_roi_padded, 0, 0);
                else CV_Error(cv::Error::StsBadArg, "Invalid type");

                nodes_private[0]->in_headers.push_back(buffer_header);

                int tile_size = buffer_header.rows * buffer_header.cols;
                int type_size = CvTypeSize(in.type());
                int footprint = tile_size * type_size;
                buffer_cur += footprint;
                buffer_cur += aligned256_after_n_char(buffer_cur);
            }
        }
        { // assign nodes' in/out headers, except first in and last out
            for(int nodes_iter = 1; nodes_iter < nodes_private.size(); nodes_iter++){
                auto& cur_node = nodes_private[nodes_iter];
                char* buffer_cur = filter_buffer[cur_node->which_buffer];
                cur_node->in_headers.clear();
                for(int in_iter=0; in_iter<cur_node->input_num; in_iter++){
                    // data is not owned by mat if initialize it this way
                    cv::Mat buffer_header(cur_roi.height + cur_node->padded_row * 2,
                                          cur_roi.width + cur_node->padded_col * 2, cur_node->input_type, buffer_cur);
                    cur_node->in_headers.push_back(buffer_header);

                    int tile_size = buffer_header.rows * buffer_header.cols;
                    int type_size = CvTypeSize(buffer_header.type());
                    int footprint = tile_size * type_size;
                    buffer_cur += footprint;
                    buffer_cur += aligned256_after_n_char(buffer_cur);
                }
                nodes_private[nodes_iter - 1]->out_headers = cur_node->in_headers;
            }
        }
        { // link last out to out_v
            if(nodes_private.back()->op_name != "linearizeTxT"){ // linearize is a very special operations
                nodes_private.back()->out_headers.clear();
                for(auto& out_ori: out_v){
                    cv::Mat out = out_ori(cur_roi);
                    nodes_private.back()->out_headers.push_back(out);
                }
            }else{ // assign global row / col to node
                nodes_private.back()->cur_row = cur_roi.y;
                nodes_private.back()->cur_col = cur_roi.x;
            }
        }
        { // some node may have special headers to link
            for(auto& node: nodes_private){
                if(node->have_special_headers){
                    node->link_special_header(cur_roi);
                }
            }
        }

        // update one by one
        for(int i=0; i<nodes_private.size(); i++) nodes_private[i]->update();
    }
    delete[] buffer_0;
    delete[] buffer_1;
#ifdef _OPENMP
    }
#endif

    }
    bool check_if_nodes_valid(){
        bool is_valid = true;
        for(int i=0; i<nodes_.size()-1; i++){
            auto& cur_node = nodes_[i];
            auto& next_node = nodes_[i+1];
            if(cur_node->output_num != next_node->input_num ||
                    cur_node->output_type != next_node->input_type){
                std::cout << "node " << i << "is not compatible with node" << i+1 << std::endl;
                is_valid = false;
            }
        }
        return is_valid;
    }

    std::vector<cv::Rect> update_rois_;
    std::vector<std::shared_ptr<FilterNode>> nodes_;
    int tileRows_, tileCols_;
    int num_threads_ = 4;
    int maxMemoFootprint_;
};

}


