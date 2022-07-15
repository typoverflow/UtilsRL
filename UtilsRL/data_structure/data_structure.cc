#include <pybind11/pybind11.h>
#include <stdexcept>
#include <vector>
#include <cmath>

#include <iostream>

namespace py = pybind11;
using namespace pybind11::literals;
using namespace std;

#include "sumtree.h"

PYBIND11_MODULE(data_structure, m) {
    py::class_<SumTree>(m, "SumTree")
        .def(py::init<int>())
        .def("update", &SumTree::update)
        .def("add", &SumTree::add)
        .def("find", &SumTree::find);
}