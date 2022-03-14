#include "makeProcess.h"
#include "forLoop.h"

namespace py = pybind11;
using namespace pybind11::literals;


void makeAllProcessSO3(py::module &m) { 
    forSOVec( [&m](auto A, auto V){
        std::string nameSO3 = "SO3"+makeNameSO<A>() + "_Vec" + std::to_string(V);
        makeProcess<InEKF::SO3<A>,Eigen::Matrix<double,V,1>>(m, nameSO3);
    });

    forSO( [&m](auto A){
        std::string nameSO3 = "SO3"+makeNameSO<A>() + "_SO3"+makeNameSO<A>();
        makeProcess<InEKF::SO3<A>,InEKF::SO3<A>>(m, nameSO3);
    });
 }