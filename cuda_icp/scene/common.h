#pragma once

// common function frequently used by others

#include "../geometry.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

#ifdef CUDA_ON
// thrust device vector can't be used in cpp by design
// same codes in cuda renderer,
// because we don't want these two related to each other
template <typename T>
class device_vector_holder{
public:
    T* __gpu_memory;
    size_t __size;
    bool valid = false;
    device_vector_holder(){}
    device_vector_holder(size_t size);
    device_vector_holder(size_t size, T init);
    ~device_vector_holder();

    T* data(){return __gpu_memory;}
    thrust::device_ptr<T> data_thr(){return thrust::device_ptr<T>(__gpu_memory);}
    T* begin(){return __gpu_memory;}
    thrust::device_ptr<T> begin_thr(){return thrust::device_ptr<T>(__gpu_memory);}
    T* end(){return __gpu_memory + __size;}
    thrust::device_ptr<T> end_thr(){return thrust::device_ptr<T>(__gpu_memory + __size);}

    size_t size(){return __size;}

    void __malloc(size_t size);
    void __free();
};

extern template class device_vector_holder<Vec2f>;
#endif

namespace songyuncen_subpixel {
// https://github.com/songyuncen/EdgesSubPix/blob/master/EdgesSubPix.cpp

    double getAmplitude(cv::Mat &dx, cv::Mat &dy, int i, int j);
    void getMagNeighbourhood(cv::Mat &dx, cv::Mat &dy, cv::Point &p,
                             int w, int h, std::vector<double> &mag);
    void get2ndFacetModelIn3x3(std::vector<double> &mag, std::vector<double> &a);
    void eigenvals(std::vector<double> &a, double eigval[2], double eigvec[2][2]);
}

template<class T>
inline T pow2(const T& in){return in*in;}
