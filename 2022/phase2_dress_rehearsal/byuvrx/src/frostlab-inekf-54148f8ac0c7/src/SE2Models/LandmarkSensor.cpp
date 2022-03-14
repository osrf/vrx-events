#include "SE2Models/LandmarkSensor.h"

namespace InEKF {

LandmarkSensor::LandmarkSensor(double std_r, double std_b) {
    error_ = ERROR::RIGHT;
    M_rb = Eigen::Matrix2d::Identity();
    M_rb(0,0) = std_r*std_r;
    M_rb(1,1) = std_b*std_b;

    HSmall.block<2,2>(0,0) = -1*MatrixS::Identity();
    HSmall.block<2,2>(0,2) =    MatrixS::Identity();
}

void LandmarkSensor::sawLandmark(int idx, const SE2<Eigen::Dynamic>& state){
    lmIdx = idx;
    b_ = Eigen::VectorXd::Zero(state().cols());
    b_(2) = 1;
    b_(idx+3) = -1;

    int rSize = 2;
    int rDim  = 1;

    int curr_cols = state().cols() - rSize;
    int curr_dim  = rDim + rSize*curr_cols; 
    this->H_ = MatrixH::Zero(rSize, curr_dim);
    this->H_.block(0, 1, rSize, rSize) = -1*MatrixS::Identity();
    this->H_.block(0, rSize*(idx+1)+rDim, rSize, rSize) = MatrixS::Identity();
}

LandmarkSensor::VectorB LandmarkSensor::processZ(const Eigen::VectorXd& z, const SE2<Eigen::Dynamic>& state){
    // convert range and bearing into x and y
    if(z.rows() == 2){
        double r = z(0);
        double b = z(1);
        VectorB z_ = b_;
        z_(0) = r*cos(b);
        z_(1) = r*sin(b);

        // convert r/b cov -> x/y cov
        MatrixS G;
        G << cos(b), -r*sin(b),
            sin(b), r*cos(b);
        this->M_.noalias() = G*M_rb*G.transpose();

        return z_;
    }

    else{
        return z;
    }
}

LandmarkSensor::MatrixS LandmarkSensor::calcSInverse(const SE2<Eigen::Dynamic>& state){
    MatrixS Sinv;
    MatrixS R = state.R()();
    // if we're using a right filter, H_error_ will be identities
    // So take out the parts that will be 0 so we can speed things up
    if(filterError == ERROR::RIGHT){
        Eigen::Matrix4d CovSliced;
        CovSliced.block<2,2>(0,0) = state.Cov().block<2,2>(1,1);
        CovSliced.block<2,2>(2,2) = state.Cov().block<2,2>(lmIdx*2+3,lmIdx*2+3);
        CovSliced.block<2,2>(0,2) = state.Cov().block<2,2>(1,lmIdx*2+3);
        CovSliced.block<2,2>(2,0) = state.Cov().block<2,2>(lmIdx*2+3,1);
        // Eigen::Matrix4d CovSliced = state.Cov()({1,2,lmIdx*2+3,lmIdx*2+4},{1,2,lmIdx*2+3,lmIdx*2+4});
        Sinv.noalias() = ( HSmall*CovSliced*HSmall.transpose() + R*M_*R.transpose() ).inverse();
    }
    else{
        Sinv.noalias() = ( H_error_*state.Cov()*H_error_.transpose() + R*M_*R.transpose() ).inverse();
    }

    return Sinv;
}

double LandmarkSensor::calcMahDist(const Eigen::VectorXd& z, const SE2<Eigen::Dynamic>& state){
    makeHError(state, filterError);
    VectorB z_ = processZ(z, state);
    VectorV V  = calcV(z_, state);
    MatrixS Sinv = calcSInverse(state);
    return V.transpose()*Sinv*V;
}

}