#include "makeInEKF.h"
#include "forLoop.h"

namespace py = pybind11;
using namespace pybind11::literals;


void makeAllInEKFSO3(py::module &m) { 
    forSOVec( [&m](auto A, auto V){
        std::string nameSO3 = "SO3"+makeNameSO<A>() + "_Vec" + std::to_string(V);
        makeInEKF<InEKF::SO3<A>,Eigen::Matrix<double,V,1>>(m, nameSO3);
    });

    forSO( [&m](auto A){
        std::string nameSO3 = "SO3"+makeNameSO<A>() + "_SO3"+makeNameSO<A>();
        makeInEKF<InEKF::SO3<A>,InEKF::SO3<A>>(m, nameSO3);
    });
 }