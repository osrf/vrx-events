#ifndef INERTIAL_PROCESS
#define INERTIAL_PROCESS

#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>
#include <InEKF/Core>

namespace Eigen{
    typedef Matrix<double,6,1> Vector6d;
}

namespace InEKF {

class InertialProcess : public ProcessModel<SE3<2,6>, Eigen::Matrix<double,6,1>> {

    private:
        const Eigen::Vector3d g_ = (Eigen::Vector3d() << 0,0,-9.81).finished();

    public:
        InertialProcess();
        ~InertialProcess(){}
        SE3<2,6> f(Eigen::Vector6d u, double dt, SE3<2,6> state) override;
        MatrixCov MakePhi(const Eigen::Vector6d& u, double dt, const SE3<2,6>& state, ERROR error) override;
        
        void setGyroNoise(double std);
        void setAccelNoise(double std);
        void setGyroBiasNoise(double std);
        void setAccelBiasNoise(double std);
        

};

}

#endif // INERTIAL_PROCESS