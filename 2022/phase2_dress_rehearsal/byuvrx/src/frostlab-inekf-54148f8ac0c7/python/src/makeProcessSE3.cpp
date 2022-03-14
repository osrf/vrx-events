#include "makeProcess.h"
#include "forLoop.h"

namespace py = pybind11;
using namespace pybind11::literals;

void makeAllProcessSE3(py::module &m) { 
    forSEVec( [&m](auto C, auto A, auto V){
        std::string nameSE3 = "SE3"+makeNameSE<C,A>() + "_Vec" + std::to_string(V);
        makeProcess<InEKF::SE3<C,A>,Eigen::Matrix<double,V,1>>(m, nameSE3);
    });

    forSE( [&m](auto C, auto A){
        std::string nameSE3 = "SE3"+makeNameSE<C,A>() + "_SE3"+makeNameSE<C,A>();
        makeProcess<InEKF::SE3<C,A>,InEKF::SE3<C,A>>(m, nameSE3);
    });
 }