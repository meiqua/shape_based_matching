#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "np2mat/ndarray_converter.h"
#include "line2Dup.h"
namespace py = pybind11;

PYBIND11_MODULE(shape_based_matching_py, m) {
    NDArrayConverter::init_numpy();
    
    py::class_<line2Dup::Match>(m,"Match")
            .def(py::init<>())
            .def_readwrite("x",&line2Dup::Match::x)
            .def_readwrite("y",&line2Dup::Match::y)
            .def_readwrite("similarity",&line2Dup::Match::similarity)
            .def_readwrite("class_id",&line2Dup::Match::class_id)
            .def_readwrite("template_id",&line2Dup::Match::template_id);

    py::class_<line2Dup::Feature>(m,"Feature")
            .def(py::init<>())
            .def_readwrite("x",&line2Dup::Feature::x)
            .def_readwrite("y",&line2Dup::Feature::y)
            .def_readwrite("label",&line2Dup::Feature::label);

    py::class_<line2Dup::Template>(m,"Template")
            .def(py::init<>())
            .def_readwrite("width",&line2Dup::Template::width)
            .def_readwrite("height",&line2Dup::Template::height)
            .def_readwrite("tl_x",&line2Dup::Template::tl_x)
            .def_readwrite("tl_y",&line2Dup::Template::tl_y)
            .def_readwrite("pyramid_level",&line2Dup::Template::pyramid_level)
            .def_readwrite("features",&line2Dup::Template::features);

    py::class_<cv::Point2f>(m,"CV_Point2f")
            .def(py::init<>())
            .def(py::init<float, float>());

    py::class_<line2Dup::Detector>(m, "Detector")
        .def(py::init<>())
        .def(py::init<int, std::vector<int>, float, float>(),
            py::arg("num_features"), py::arg("spread_and_pyr"),
            py::arg("min_det_contrast") = 30, py::arg("min_train_contrast")=60)
        .def("addTemplate", &line2Dup::Detector::addTemplate, py::arg("sources"), py::arg("class_id"),
             py::arg("object_mask") = cv::Mat(), py::arg("num_features") = 0)
        .def("addTemplate_rotate", &line2Dup::Detector::addTemplate_rotate, 
            py::arg("class_id"), py::arg("zero_id"), py::arg("theta"), py::arg("center"))
        .def("writeClasses", &line2Dup::Detector::writeClasses)
        .def("clear_classes", &line2Dup::Detector::clear_classes)
        .def("readClasses", &line2Dup::Detector::readClasses)
        .def("match", &line2Dup::Detector::match, py::arg("sources"),
             py::arg("threshold"), py::arg("class_ids"), py::arg("masks")=cv::Mat())
        .def("getTemplates", &line2Dup::Detector::getTemplates);

    py::class_<shape_based_matching::Info>(m, "Info")
        .def(py::init<>())
        .def(py::init<float, float>())
        .def_readwrite("angle",&shape_based_matching::Info::angle)
        .def_readwrite("scale",&shape_based_matching::Info::scale);

        py::class_<shape_based_matching::shapeInfo_producer>(m, "shapeInfo_producer")
        .def(py::init<>())
        .def(py::init<cv::Mat, cv::Mat>())
        .def_readwrite("infos",&shape_based_matching::shapeInfo_producer::infos)
        .def_readwrite("angle_range",&shape_based_matching::shapeInfo_producer::angle_range)
        .def_readwrite("angle_step",&shape_based_matching::shapeInfo_producer::angle_step)
        .def_readwrite("scale_range",&shape_based_matching::shapeInfo_producer::scale_range)
        .def_readwrite("scale_step",&shape_based_matching::shapeInfo_producer::scale_step)
        .def("produce_infos", &shape_based_matching::shapeInfo_producer::produce_infos)
        .def("save_infos", &shape_based_matching::shapeInfo_producer::save_infos,
            py::arg("infos"), py::arg("path"))
        .def("load_infos", &shape_based_matching::shapeInfo_producer::load_infos,
            py::arg("path"))
        .def("src_of", &shape_based_matching::shapeInfo_producer::src_of,
            py::arg("info"))
        .def("mask_of", &shape_based_matching::shapeInfo_producer::mask_of,
            py::arg("info"));
}
