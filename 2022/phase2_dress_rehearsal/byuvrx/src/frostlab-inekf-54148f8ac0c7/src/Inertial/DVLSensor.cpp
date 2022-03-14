#include "Inertial/DVLSensor.h"

namespace InEKF {

DVLSensor::DVLSensor(SO3<> dvlR, Eigen::Vector3d dvlT)
    : dvlR_(dvlR), dvlT_(SO3<>::Wedge(dvlT)){
    error_ = ERROR::RIGHT;

    M_ = Eigen::Matrix3d::Zero();
    H_ = Eigen::Matrix<double, 3, 15>::Zero();
    H_.block<3,3>(0,3) = Eigen::Matrix3d::Identity();
}

DVLSensor::DVLSensor(SE3<> dvlH)
    : DVLSensor(dvlH.R(), dvlH[0]) {}

DVLSensor::DVLSensor(Eigen::Matrix3d dvlR, Eigen::Vector3d dvlT)
    : DVLSensor(SO3<>(dvlR), dvlT) {}

DVLSensor::DVLSensor() 
    : DVLSensor(SO3<>(), Eigen::Vector3d::Zero()) {
}

void DVLSensor::setNoise(double std_dvl, double std_imu){
    M_ = Eigen::Matrix3d::Identity() * std_dvl*std_dvl;
    
    // Rotate into IMU frame
    Eigen::Matrix3d IMU = Eigen::Matrix3d::Identity() * std_imu*std_imu;
    M_ = dvlR_()*M_*dvlR_().transpose() + dvlT_*IMU*dvlT_.transpose();
}

DVLSensor::VectorB DVLSensor::processZ(const Eigen::VectorXd& z, const SE3<2,6>& state){
    // Fill up Z
    Eigen::Matrix<double,5,1> z_full;
    z_full << z[0], z[1], z[2], -1, 0;

    // Convert to IMU frame
    Eigen::Vector3d omega = z.tail(3);
    z_full.head(3) = dvlR_()*z_full.head(3) + dvlT_*omega;

    return z_full;
}

}