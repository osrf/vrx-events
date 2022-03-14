#include "SE2Models/OdometryProcessDynamic.h"

namespace InEKF {


SE2<Eigen::Dynamic> OdometryProcessDynamic::f(SE2<> u, double dt, SE2<Eigen::Dynamic> state){
    SE2<Eigen::Dynamic>::MatrixState s = state();
    s.block<3,3>(0,0) = state().block<3,3>(0,0) * u();
    state.setState(s);
    return state;
}

typedef typename SE2<Eigen::Dynamic>::MatrixCov MatrixCov;
MatrixCov OdometryProcessDynamic::MakePhi(const SE2<>& u, double dt, const SE2<Eigen::Dynamic>& state, ERROR error){
    // For updating, we'll need Q to be full sized as well
    if(Q_.cols() != state.Cov().cols()){
        MatrixCov Q_temp = MatrixCov::Zero(state.Cov().cols(), state.Cov().rows());
        Q_temp.block<3,3>(0,0) = Q_.block<3,3>(0,0);
        Q_ = Q_temp;
    }

    if(error == ERROR::RIGHT){
        int curr_cols = state().cols() - 2;
        int curr_dim  = 1 + curr_cols*2;
        return MatrixCov::Identity(curr_dim,curr_dim);
    }
    else{
        SE2<Eigen::Dynamic>::MatrixState temp = SE2<Eigen::Dynamic>::MatrixState::Identity(state().cols(), state().cols());
        temp.block<3,3>(0,0) = u();
        SE2<Eigen::Dynamic> big_u(temp);
        return big_u.Ad();
    }
}

}