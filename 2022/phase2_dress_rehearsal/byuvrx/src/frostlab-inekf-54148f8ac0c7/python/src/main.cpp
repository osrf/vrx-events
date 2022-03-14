#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/operators.h>

#include "makeLie.h"
#include "makeMeasure.h"
#include "makeProcess.h"
#include "makeInEKF.h"

// Definitions found in other cpp files
// These are for loops to loop over all templates
// Split up into files to help divide compiling
void makeAllInEKFSE2(py::module &m);
void makeAllSO(py::module &m);
void makeAllSE(py::module &m);
void makeAllMeasureSO(py::module &m);
void makeAllMeasureSE(py::module &m);
void makeAllProcessSO2(py::module &m);
void makeAllProcessSO3(py::module &m);
void makeAllProcessSE2(py::module &m);
void makeAllProcessSE3(py::module &m);
void makeAllInEKFSO2(py::module &m);
void makeAllInEKFSO3(py::module &m);
void makeAllInEKFSE3(py::module &m);

void makeInertial(py::module &m);
void makeSE2Models(py::module &m);

PYBIND11_MODULE(_inekf, m) {
    m.doc() = "Invariant Extended Kalman Filter"; // optional module docstring

    py::enum_<InEKF::ERROR>(m, "ERROR")
        .value("RIGHT", InEKF::RIGHT)
        .value("LEFT", InEKF::LEFT);

    // The amount each of these instantiate is found in globals.cpp
    // If only need specific templates for a smaller python binary, turn these off and select what you need below
    makeAllSO(m);
    makeAllSE(m);

    makeAllMeasureSO(m);
    makeAllMeasureSE(m);

    makeAllProcessSO2(m);
    makeAllProcessSO3(m);
    makeAllProcessSE2(m);
    makeAllProcessSE3(m);

    makeAllInEKFSO2(m);
    makeAllInEKFSO3(m);
    makeAllInEKFSE2(m);
    makeAllInEKFSE3(m);

    makeInertial(m);
    makeSE2Models(m);

    // If more templates of any kind are needed, use these functions
    // These template functions found in headers
    // makeSO2<11>(m);
    // makeSO3<12>(m);
    // makeSE2<13,14>(m);
    // makeSE3<15,16>(m);
    // makeMeasure<InEKF::SO2<1>>(m, "SO2_1");
    // makeGenericMeasure<InEKF::SO2<1>>(m, "SO2_1");
    // makeProcess<InEKF::SO2<1>,Eigen::Vector5d>(m, "SO2_Vec5");
    // makeInEKF<InEKF::SE3<2,6>,Eigen::Matrix<double,6,1>>(m, "SE3_2_6_Vec6");
}