#include "makeProcess.h"
#include "forLoop.h"

namespace py = pybind11;
using namespace pybind11::literals;


void makeAllProcessSE2(py::module &m) { 
    forSEVec( [&m](auto C, auto A, auto V){
        std::string nameSE2 = "SE2"+makeNameSE<C,A>() + "_Vec" + std::to_string(V);
        makeProcess<InEKF::SE2<C,A>,Eigen::Matrix<double,V,1>>(m, nameSE2);
    });

    forSE( [&m](auto C, auto A){
        std::string nameSE2 = "SE2"+makeNameSE<C,A>() + "_SE2"+makeNameSE<C,A>();
        makeProcess<InEKF::SE2<C,A>,InEKF::SE2<C,A>>(m, nameSE2);
    });
 }