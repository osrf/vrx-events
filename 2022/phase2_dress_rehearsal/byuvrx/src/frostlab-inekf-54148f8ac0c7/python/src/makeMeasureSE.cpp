#include "makeMeasure.h"
#include "forLoop.h"

void makeAllMeasureSE(py::module &m) { 
    forSE( [&m](auto C, auto A){
        makeMeasure<InEKF::SE2<C,A>>(m, "SE2"+makeNameSE<C,A>());
        makeMeasure<InEKF::SE3<C,A>>(m, "SE3"+makeNameSE<C,A>());

        makeGenericMeasure<InEKF::SE2<C,A>>(m, "SE2"+makeNameSE<C,A>());
        makeGenericMeasure<InEKF::SE3<C,A>>(m, "SE3"+makeNameSE<C,A>());
    });
}