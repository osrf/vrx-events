#include "SE2Models/GPSSensor.h"

namespace InEKF {

GPSSensor::GPSSensor(double std) {
    error_ = ERROR::LEFT;
    M_ = Eigen::Matrix2d::Identity() *std*std;
}

GPSSensor::VectorB GPSSensor::processZ(const Eigen::VectorXd& z, const SE2<Eigen::Dynamic>& state){
    int curr_cols = state().cols() - 2;
    int curr_dim  = 1 + 2*curr_cols; 
    H_ = MatrixH::Zero(2, curr_dim);
    H_.block<2,2>(0,1) = Eigen::Matrix2d::Identity();

    if(z.rows() == 2){
        VectorB z_ = Eigen::VectorXd::Zero(state().cols());
        z_(0) = z(0);
        z_(1) = z(1);
        z_(2) = 1;

        return z_;
    }
    else{
        return z;
    }
}

}