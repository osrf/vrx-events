#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/operators.h>

#include <string>

#include <Eigen/Core>

#include <InEKF/Core>
#include <InEKF/SE2Models>

#include "makeInEKF.h"
#include "makeProcess.h"

namespace py = pybind11;
using namespace pybind11::literals;

void makeSE2Models(py::module &m){
    // OdometryProcess
    py::class_<InEKF::OdometryProcess, InEKF::ProcessModel<InEKF::SE2<>, InEKF::SE2<>>, std::shared_ptr<InEKF::OdometryProcess>>(m, "OdometryProcess")
        .def(py::init<>())
        .def("setQ", py::overload_cast<double>(&InEKF::OdometryProcess::setQ),
            "q"_a)
        .def("setQ", py::overload_cast<Eigen::Vector3d>(&InEKF::OdometryProcess::setQ),
            "q"_a)
        .def("setQ", py::overload_cast<Eigen::Matrix3d>(&InEKF::OdometryProcess::setQ),
            "q"_a);


    makeProcess<InEKF::SE2<Eigen::Dynamic>, InEKF::SE2<>>(m, "SE2_D_0_SE2_1_0");
    makeInEKF<InEKF::SE2<Eigen::Dynamic>, InEKF::SE2<>>(m, "SE2_D_0_SE2_1_0");
 
    using BaseProcess = InEKF::ProcessModel<InEKF::SE2<Eigen::Dynamic>, InEKF::SE2<>>;
    using BaseMeasure = InEKF::MeasureModel<InEKF::SE2<Eigen::Dynamic>>;
    // NOTE: Don't have to put in overriden functions if include base classes
    // Just have to put in new classes

    // OdometryProcessDynamic
    py::class_<InEKF::OdometryProcessDynamic, BaseProcess, std::shared_ptr<InEKF::OdometryProcessDynamic>>(m, "OdometryProcessDynamic")
        .def(py::init<>())
        .def("setQ", py::overload_cast<double>(&InEKF::OdometryProcessDynamic::setQ),
            "q"_a)
        .def("setQ", py::overload_cast<Eigen::Vector3d>(&InEKF::OdometryProcessDynamic::setQ),
            "q"_a)
        .def("setQ", py::overload_cast<Eigen::Matrix3d>(&InEKF::OdometryProcessDynamic::setQ),
            "q"_a);

    // Landmark Sensor
    py::class_<InEKF::LandmarkSensor, BaseMeasure>(m, "LandmarkSensor")
        .def(py::init<double, double>(), "std_r"_a, "std_b"_a)
        .def("sawLandmark", &InEKF::LandmarkSensor::sawLandmark)
        .def("calcMahDist", &InEKF::LandmarkSensor::calcMahDist);
        
    // Landmark Sensor
    py::class_<InEKF::GPSSensor, BaseMeasure>(m, "GPSSensor")
        .def(py::init<double>(), "std"_a);
}