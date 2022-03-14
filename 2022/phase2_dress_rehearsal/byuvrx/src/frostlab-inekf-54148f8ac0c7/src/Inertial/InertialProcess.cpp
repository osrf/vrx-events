#include "Inertial/InertialProcess.h"

namespace InEKF {

InertialProcess::InertialProcess() {
    Q_ = MatrixCov::Zero();
}

SE3<2,6> InertialProcess::f(Eigen::Vector6d u, double dt, SE3<2,6> state){
    // Get everything we need
    Eigen::Vector6d u_shifted = u - state.Aug();
    Eigen::Vector3d omega = u_shifted.head(3);
    Eigen::Vector3d a = u_shifted.tail(3);
    Eigen::Matrix3d R = state.R()();
    Eigen::Vector3d v = state[0];
    Eigen::Vector3d p = state[1];

    // Calculate
    MatrixState S = MatrixState::Identity();
    S.block(0,0,3,3) = R * SO3<>::Exp(omega*dt)();
    S.block(0,3,3,1) = v + (R*a + g_)*dt;
    S.block(0,4,3,1) = p + v*dt + (R*a + g_)*dt*dt/2;

    state.setState(S);

    return state;
}

typedef typename SE3<2,6>::MatrixCov MatrixCov;
MatrixCov InertialProcess::MakePhi(const Eigen::Vector6d& u, double dt, const SE3<2,6>& state, ERROR error){
    MatrixCov A = MatrixCov::Zero();

    if(error == ERROR::RIGHT){
        // Get everything we need
        Eigen::Matrix3d R = state.R()();
        Eigen::Matrix3d v_cross = SO3<>::Wedge( state[0] );
        Eigen::Matrix3d p_cross = SO3<>::Wedge( state[1] );

        A.block<3,3>(3,0) = SO3<>::Wedge(g_);
        A.block<3,3>(6,3) = Eigen::Matrix3d::Identity();
        
        A.block<3,3>(0,9) = -R;
        A.block<3,3>(3,9) = -v_cross * R;
        A.block<3,3>(6,9) = -p_cross * R;
        A.block<3,3>(3,12) = -R;

        return MatrixCov::Identity() + A*dt + A*A*dt*dt/2 + A*A*A*dt*dt*dt/6;
    }
    else{
        Eigen::Vector6d u_shifted = u - state.Aug();
        Eigen::Matrix3d w_cross = SO3<>::Wedge( u_shifted.head(3) );
        Eigen::Matrix3d a_cross = SO3<>::Wedge( u_shifted.tail(3) );

        A.block<3,3>(0,0) = -w_cross;
        A.block<3,3>(3,3) = -w_cross;
        A.block<3,3>(6,6) = -w_cross;
        A.block<3,3>(3,0) = -a_cross;
        A.block<3,3>(6,3) = Eigen::Matrix3d::Identity();

        A.block<3,3>(0,9) = -Eigen::Matrix3d::Identity();
        A.block<3,3>(3,12) = -Eigen::Matrix3d::Identity();
        
        return (A*dt).exp();
    }

}

void InertialProcess::setGyroNoise(double std){
    Q_.block<3,3>(0,0) = Eigen::Matrix3d::Identity() * std*std;
}

void InertialProcess::setAccelNoise(double std){
    Q_.block<3,3>(3,3) = Eigen::Matrix3d::Identity() * std*std;
}

void InertialProcess::setGyroBiasNoise(double std){
    Q_.block<3,3>(9,9) = Eigen::Matrix3d::Identity() * std*std;
}

void InertialProcess::setAccelBiasNoise(double std){
    Q_.block<3,3>(12,12) = Eigen::Matrix3d::Identity() * std*std;
}

}