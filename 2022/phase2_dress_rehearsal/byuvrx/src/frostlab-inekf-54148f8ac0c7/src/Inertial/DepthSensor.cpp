#include "Inertial/DepthSensor.h"

namespace InEKF {

DepthSensor::DepthSensor(double std) {
    error_ = ERROR::LEFT;

    M_ = Eigen::Matrix3d::Zero();
    M_(2,2) = 1 / (std*std);

    H_ = Eigen::Matrix<double, 3, 15>::Zero();
    H_.block<3,3>(0,6) = Eigen::Matrix3d::Identity();
}

void DepthSensor::setNoise(double std){
    // Actually storing M.inverse() here
    M_ = Eigen::Matrix3d::Zero();
    M_(2,2) = 1 / (std*std);
}

DepthSensor::VectorB DepthSensor::processZ(const Eigen::VectorXd& z, const SE3<2,6>& state) {
    Eigen::Matrix<double,5,1> z_full;
    Eigen::Vector3d p = state[1];
    z_full << p[0], p[1], z[0], 0, 1;
    return z_full;
}

DepthSensor::MatrixS DepthSensor::calcSInverse(const SE3<2,6>& state){
    // Calculate Sinv
    Eigen::Matrix3d R = state.R()();
    Eigen::Matrix3d Sig = (H_error_ * state.Cov() * H_error_.transpose()).inverse();
    return Sig - Sig*( R.transpose()*M_*R + Sig ).inverse()*Sig;
}

}