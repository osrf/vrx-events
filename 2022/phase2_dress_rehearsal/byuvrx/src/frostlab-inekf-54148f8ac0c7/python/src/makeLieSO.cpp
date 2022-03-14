#include "makeLie.h"
#include "forLoop.h"

void makeAllSO(py::module &m) { 
    forSO( [&m](auto i){
        makeSO2<i>(m);
        makeSO3<i>(m);
    } );
}
