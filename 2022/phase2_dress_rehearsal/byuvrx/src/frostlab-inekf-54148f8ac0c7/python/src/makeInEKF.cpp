#include "InEKF.h"
#include "globals.h"

namespace py = pybind11;
using namespace pybind11::literals;

// mini for loop for iterating over vectors
template<class G, int V>
struct forLoopInEKFVector {
    static void value(py::module &m, std::string name){
        std::string myName = name + "_Vec" + std::to_string(V);
        makeInEKF<G,Eigen::Matrix<double,V,1>>(m,myName);

        forLoopInEKFVector<G,V-1>::value(m, name);
    }
};
template<class G>
struct forLoopInEKFVector<G,0> {
    static void value(py::module &m, std::string name){}
};

// Compile time double for loop
template <int C, int A>
struct forLoopInEKF {
    static void value(py::module &m){
        std::string nameSE2 = "SE2"+makeNameSE<C,A>() + "_SE2"+makeNameSE<C,A>();
        makeInEKF<InEKF::SE2<C,A>,InEKF::SE2<C,A>>(m, nameSE2);
        // std::string nameSE3 = "SE3"+makeNameSE<C,A>() + "_SE3"+makeNameSE<C,A>();
        // makeInEKF<InEKF::SE3<C,A>,InEKF::SE3<C,A>>(m, nameSE3);

        forLoopInEKFVector<InEKF::SE2<C,A>, VEC>::value(m, "SE2"+makeNameSE<C,A>());
        // forLoopInEKFVector<InEKF::SE3<C,A>, VEC>::value(m, "SE3"+makeNameSE<C,A>());

        forLoopInEKF<C, A-1>::value(m);
    }
};

// Skip 0's for Col
template<int A>
struct forLoopInEKF<0,A>{
    static void value(py::module &m){
        forLoopInEKF<-1,A>::value(m);
    }
};

// when we hit -2 for aug, flip back to top
template <int C>
struct forLoopInEKF<C,-2>{
    static void value(py::module &m){
        forLoopInEKF<C-1,AUG>::value(m);
    }
};

// Also do SO2/3 when C=-2
template <int A>
struct forLoopInEKF<-2,A> {
    static void value(py::module &m){
        // std::string nameSO2 = "SO2"+makeNameSO<A>() + "_SO2"+makeNameSO<A>();
        // makeInEKF<InEKF::SO2<A>,InEKF::SO2<A>>(m, nameSO2);
        // std::string nameSO3 = "SO3"+makeNameSO<A>() + "_SO3"+makeNameSO<A>();
        // makeInEKF<InEKF::SO3<A>,InEKF::SO3<A>>(m, nameSO3);

        // forLoopInEKFVector<InEKF::SO2<A>, VEC>::value(m, "SO2"+makeNameSO<A>());
        // forLoopInEKFVector<InEKF::SO3<A>, VEC>::value(m, "SO3"+makeNameSO<A>());

        // forLoopInEKF<-2, A-1>::value(m);
    }
};

// ending condition
template <>
struct forLoopInEKF<-2,-2> {
    static void value(py::module &m){}
};


void makeAllInEKF(py::module &m) { forLoopInEKF<COL,AUG>::value(m); }