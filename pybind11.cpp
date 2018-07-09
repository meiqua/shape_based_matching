#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "np2mat/ndarray_converter.h"
#include "line2Dup.h"
namespace py = pybind11;

PYBIND11_MODULE(line2Dup_pybind, m) {
    NDArrayConverter::init_numpy();

    py::class_<line2Dup::Match>(m,"Match")
            .def(py::init<>())
            .def_readwrite("x",&line2Dup::Match::x)
            .def_readwrite("y",&line2Dup::Match::y)
            .def_readwrite("similarity",&line2Dup::Match::similarity)
            .def_readwrite("class_id",&line2Dup::Match::class_id)
            .def_readwrite("template_id",&line2Dup::Match::template_id);


    py::class_<line2Dup::Detector>(m, "Detector")
        .def(py::init<>())
        .def(py::init<std::vector<int> >())
        .def(py::init<int, std::vector<int> >())
        .def("addTemplate", &line2Dup::Detector::addTemplate)
        .def("writeClasses", &line2Dup::Detector::writeClasses)
        .def("readClasses", &line2Dup::Detector::readClasses)
        .def("match", &line2Dup::Detector::match, py::arg("sources"),
             py::arg("threshold"), py::arg("class_ids"), py::arg("masks")=cv::Mat())
        .def("getTemplates", &line2Dup::Detector::getTemplates);
}
