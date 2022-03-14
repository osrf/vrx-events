#include "makeLie.h"
#include "forLoop.h"

namespace py = pybind11;
using namespace pybind11::literals;

void makeAllSE(py::module &m) { 
    forSE( [&m](auto i, auto j){ 
        makeSE2<i,j>(m);
        makeSE3<i,j>(m);
    } );
}