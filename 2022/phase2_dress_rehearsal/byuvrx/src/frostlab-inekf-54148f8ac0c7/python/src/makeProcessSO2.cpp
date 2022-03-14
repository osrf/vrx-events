#include "makeProcess.h"
#include "forLoop.h"

namespace py = pybind11;
using namespace pybind11::literals;


void makeAllProcessSO2(py::module &m) { 
    forSOVec( [&m](auto A, auto V){
        std::string nameSO2 = "SO2"+ makeNameSO<A>() + "_Vec" + std::to_string(V);
        makeProcess<InEKF::SO2<A>,Eigen::Matrix<double,V,1>>(m, nameSO2);
    });

    forSO( [&m](auto A){
        std::string nameSO2 = "SO2"+makeNameSO<A>() + "_SO2"+makeNameSO<A>();
        makeProcess<InEKF::SO2<A>,InEKF::SO2<A>>(m, nameSO2);
    });
 }