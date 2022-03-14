#include "SE2Models/OdometryProcess.h"

namespace InEKF {


SE2<> OdometryProcess::f(SE2<> u, double dt, SE2<> state){
    return state.compose(u);
}

typedef typename SE2<>::MatrixCov MatrixCov;
MatrixCov OdometryProcess::MakePhi(const SE2<>& u, double dt, const SE2<>& state, ERROR error){
    if(error == ERROR::RIGHT){
        return MatrixCov::Identity();
    }
    else{
        return SE2<>::Ad(u.inverse());
    }
}

}