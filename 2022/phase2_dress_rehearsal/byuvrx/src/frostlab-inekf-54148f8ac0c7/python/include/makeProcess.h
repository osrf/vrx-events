#ifndef PYTHON_PROCESSMODEL
#define PYTHON_PROCESSMODEL

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/operators.h>

#include <string>

#include <Eigen/Core>
#include <InEKF/Core>

namespace py = pybind11;
using namespace pybind11::literals;

template <class Group, class U>
class PyProcessModel : public InEKF::ProcessModel<Group,U> {
    public:
        // Use constructors
        using InEKF::ProcessModel<Group,U>::ProcessModel;
        typedef typename InEKF::ProcessModel<Group,U>::MatrixCov MatrixCov;
        typedef typename InEKF::ProcessModel<Group,U>::MatrixState MatrixState;
        // Need to typedef this b/c comma doesn't do well in the pybind macro
        typedef typename InEKF::ProcessModel<Group,U> BaseClass;

        Group f(U u, double dt, Group state) override {
            PYBIND11_OVERRIDE_PURE(
                Group,       /* Return type */
                BaseClass,   /* Parent class */
                f,           /* Name of function in C++ (must match Python name) */
                u, dt, state /* Argument(s) */
            );
        }

        MatrixCov MakePhi(const U& u, double dt, const Group& state, InEKF::ERROR error) override {
            PYBIND11_OVERRIDE_PURE(
                MatrixCov,     
                BaseClass,
                MakePhi,
                u, dt, state, error
            );
        }
};

template<class G, class U>
void makeProcess(py::module &m, std::string name){
    using K = InEKF::ProcessModel<G,U>;

    std::string namePM = "ProcessModel_" + name;
    py::class_<K, PyProcessModel<G,U>, std::shared_ptr<K>> myClass(m, namePM.c_str());
    myClass
        .def(py::init<>())
        // Overrideable methods
        .def("f", &K::f,
            "u"_a, "dt"_a, "state"_a)
        .def("MakePhi", &K::MakePhi,
            "u"_a, "dt"_a, "state"_a, "error"_a)

        // Properties
        .def_property("Q", &K::getQ, &K::setQ);
}

#endif // PYTHON_PROCESSMODEL