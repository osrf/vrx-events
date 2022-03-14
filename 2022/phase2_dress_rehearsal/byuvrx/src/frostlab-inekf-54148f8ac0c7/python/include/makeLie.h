#ifndef PYTHON_LIE
#define PYTHON_LIE

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/operators.h>

#include <string>

#include <Eigen/Core>
#include <InEKF/Core>

namespace py = pybind11;
using namespace pybind11::literals;

template <class T>
py::class_<T> make_group(py::module &m, std::string name, int num1, int num2=-10){
    // parse the name
    name += "_";
    name += num1 == Eigen::Dynamic ? "D" : std::to_string(num1);
    if(num2 != -10){
        name += "_"; 
        name += num2 == Eigen::Dynamic ? "D" : std::to_string(num2);
    }
    
    py::class_<T> myClass = py::class_<T>(m, name.c_str())
        // Getters
        .def("R", &T::R)
        .def_property_readonly("isUncertain", &T::Uncertain)
        .def_property("State", &T::operator(), &T::setState)
        .def_property("Cov", &T::Cov, &T::setCov)
        .def_property("Aug", &T::Aug, &T::setAug)

        // self operators
        .def("inverse", &T::inverse)
        .def("Ad", py::overload_cast<>(&T::Ad, py::const_))
        .def("log", &T::log)

        // Group action
        .def(py::self * py::self)

        // Static Operators
        .def_static("Wedge", &T::Wedge, "xi"_a)
        .def_static("Exp", &T::Exp, "xi"_a)
        .def_static("Log", &T::Exp, "g"_a)
        // Can't overload combination of static/instance methods
        .def_static("Adjoint", py::overload_cast<const T&>(&T::Ad), "g"_a)

        // Misc
        .def("addAug", &T::addAug, "x"_a, "sigma"_a=1)
        .def("__matmul__", &T::operator*, py::is_operator())
        .def("__invert__", &T::inverse)
        .def("__str__", &T::toString)
        .def("__repr__", &T::toString);

    return myClass;
}

template <int C, int A>
void makeSE2(py::module &m){
    py::class_<InEKF::SE2<C,A>> mySE2 = make_group<InEKF::SE2<C,A>>(m, "SE2", C, A);

    // Fill in SE2 Constructors
    int a = A == Eigen::Dynamic ? 0 : A;
    int c = (A == Eigen::Dynamic || C == Eigen::Dynamic) ? 3 : InEKF::calcStateDim(2,C,A);
    int ma = C == Eigen::Dynamic ? 3 : InEKF::calcStateMtxSize(2,C);
    typedef typename InEKF::SE2<C,A>::TangentVector SE2_TV;
    typedef typename InEKF::SE2<C,A>::MatrixCov SE2_MC;
    typedef typename InEKF::SE2<C,A>::MatrixState SE2_MS;
    typedef typename InEKF::SE2<C,A>::VectorAug SE2_VA;

    // Make sure tangent constructor is first. When using a dynamic type
    // a TV could it in a nxm matrix, but a matrix can't fit in a nx1 vector.
    mySE2.def(py::init<SE2_TV, SE2_MC>(),
            "xi"_a, "Cov"_a=SE2_MC::Zero(c,c))
        .def(py::init<SE2_MS, SE2_MC, SE2_VA>(), 
                "State"_a=SE2_MS::Identity(ma,ma), "Cov"_a=SE2_MC::Zero(c,c), "Aug"_a=SE2_VA::Zero(a))
        .def(py::init<InEKF::SE2<C,A> const &>())
        .def(py::init<double, double, double, SE2_MC>(),
            "theta"_a, "x"_a, "y"_a, "Cov"_a=SE2_MC::Zero(c,c))

        .def("__getitem__", &InEKF::SE2<C,A>::operator[])
        .def("addCol", &InEKF::SE2<C,A>::addCol,
            "x"_a, "sigma"_a=Eigen::Matrix2d::Identity());

}

template <int C, int A>
void makeSE3(py::module &m){
    // make pyclasses
    py::class_<InEKF::SE3<C,A>> mySE3 = make_group<InEKF::SE3<C,A>>(m, "SE3", C, A);

    // Fill in SE3 Constructors
    int a = A == Eigen::Dynamic ? 0 : A;
    int ma = C == Eigen::Dynamic ? 4 : InEKF::calcStateMtxSize(3,C);
    constexpr int c = (A == Eigen::Dynamic || C == Eigen::Dynamic) ? 6 : InEKF::calcStateDim(3,C,A);
    static constexpr int small_xi = (A == Eigen::Dynamic || C == Eigen::Dynamic) ? Eigen::Dynamic : c-3;
    typedef typename InEKF::SE3<C,A>::TangentVector SE3_TV;
    typedef typename InEKF::SE3<C,A>::MatrixCov SE3_MC;
    typedef typename InEKF::SE3<C,A>::MatrixState SE3_MS;
    typedef typename InEKF::SE3<C,A>::VectorAug SE3_VA;

    mySE3.def(py::init<SE3_TV, SE3_MC>(),
            "xi"_a, "Cov"_a=SE3_MC::Zero(c,c))
        .def(py::init<SE3_MS, SE3_MC, SE3_VA>(), 
                "State"_a=SE3_MS::Identity(ma,ma), "Cov"_a=SE3_MC::Zero(c,c), "Aug"_a=SE3_VA::Zero(a))
        .def(py::init<InEKF::SE3<C,A> const &>())
        .def(py::init<double, double, double, double, double, double, SE3_MC>(),
            "w1"_a, "w2"_a, "w3"_a, "x"_a, "y"_a, "z"_a, "Cov"_a=SE3_MC::Zero(c,c))
        // .def(py::init<InEKF::SO3<>, Eigen::Matrix<double,small_xi,1>, SE3_MC>)
                // "R"_a, "xi"_a, "Cov"_a=SE3_MC::Zero(c,c));

        .def("__getitem__", &InEKF::SE3<C,A>::operator[])
        .def("addCol", &InEKF::SE3<C,A>::addCol,
            "x"_a, "sigma"_a=Eigen::Matrix3d::Identity());
    // TODO Fix these constructors freaking out!

}

template <int A>
void makeSO2(py::module &m){
    py::class_<InEKF::SO2<A>> mySO2 = make_group<InEKF::SO2<A>>(m, "SO2", A);

    // Fill in SO2 Constructors
    int a = A == Eigen::Dynamic ? 0 : A;
    int c = A == Eigen::Dynamic ? 1 : A+1;
    typedef typename InEKF::SO2<A>::TangentVector SO2_TV;
    typedef typename InEKF::SO2<A>::MatrixCov SO2_MC;
    typedef typename InEKF::SO2<A>::MatrixState SO2_MS;
    typedef typename InEKF::SO2<A>::VectorAug SO2_VA;
    mySO2.def(py::init<SO2_MS, SO2_MC, SO2_VA>(), 
            "State"_a=SO2_MS::Identity(), "Cov"_a=SO2_MC::Zero(c,c), "Aug"_a=SO2_VA::Zero(a))
        .def(py::init<InEKF::SO2<A> const &>())
        .def(py::init<SO2_TV, SO2_MC>(),
            "xi"_a, "Cov"_a=SO2_MC::Zero(c,c))
        .def(py::init<double, SO2_MC, SO2_VA>(),
            "theta"_a, "Cov"_a=SO2_MC::Zero(c,c), "Aug"_a=SO2_VA::Zero(a));

}

template <int A>
void makeSO3(py::module &m){
    py::class_<InEKF::SO3<A>> mySO3 = make_group<InEKF::SO3<A>>(m, "SO3", A);

    // Fill in SO3 Constructors
    int a = A == Eigen::Dynamic ? 0 : A;
    int c = A == Eigen::Dynamic ? 3 : A+3;
    typedef typename InEKF::SO3<A>::TangentVector SO3_TV;
    typedef typename InEKF::SO3<A>::MatrixCov SO3_MC;
    typedef typename InEKF::SO3<A>::MatrixState SO3_MS;
    typedef typename InEKF::SO3<A>::VectorAug SO3_VA;
    mySO3.def(py::init<SO3_MS, SO3_MC, SO3_VA>(), 
            "State"_a=SO3_MS::Identity(), "Cov"_a=SO3_MC::Zero(c,c), "Aug"_a=SO3_VA::Zero(a))
        .def(py::init<InEKF::SO3<A> const &>())
        .def(py::init<SO3_TV, SO3_MC>(),
            "xi"_a, "Cov"_a=SO3_MC::Zero(c,c))
        .def(py::init<double, double, double, SO3_MC, SO3_VA>(),
            "w1"_a, "w2"_a, "w3"_a, "Cov"_a=SO3_MC::Zero(c,c), "Aug"_a=SO3_VA::Zero(a));

}


#endif // PYTHON_LIE