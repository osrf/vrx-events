#ifndef PYTHON_INEKF
#define PYTHON_INEKF

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/operators.h>

#include <string>

#include <Eigen/Core>
#include <InEKF/Core>

namespace py = pybind11;
using namespace pybind11::literals;

template<class G, class U>
void makeInEKF(py::module &m, std::string name){
    using T = InEKF::InEKF<InEKF::ProcessModel<G,U>>;
    typedef Eigen::Matrix<double,G::rotSize,G::N> MatrixH;

    name = "InEKF_" + name;
    py::class_<T> myClass(m, name.c_str());
    myClass
        .def(py::init<G, InEKF::ERROR>(),
            "state"_a, "error"_a=InEKF::RIGHT)
        
        .def("Predict", &T::Predict,
            "u"_a, "dt"_a=1)
        .def("Update", py::overload_cast<const Eigen::VectorXd&,std::string>(&T::Update),
            "z"_a, "type"_a)
        .def("Update", py::overload_cast<const Eigen::VectorXd&,std::string,MatrixH>(&T::Update),
            "z"_a, "type"_a, "H"_a)
        .def("addMeasureModel", &T::addMeasureModel,
            "name"_a, "m"_a)

        .def_readwrite("state", &T::state_)
        .def_readwrite("pModel", &T::pModel);

}

#endif // PYTHON_INEKF