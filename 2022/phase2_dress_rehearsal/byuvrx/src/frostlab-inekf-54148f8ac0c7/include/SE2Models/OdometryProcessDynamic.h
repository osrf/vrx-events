#ifndef ODOMETRY_PROCESS_DYNAMIC
#define ODOMETRY_PROCESS_DYNAMIC

#include <Eigen/Core>

#include "InEKF/Core"

namespace InEKF {

class OdometryProcessDynamic : public ProcessModel<SE2<Eigen::Dynamic>, SE2<>> {

    private:
        const Eigen::Vector3d g_;

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        OdometryProcessDynamic(){}
        ~OdometryProcessDynamic(){}
        
        SE2<Eigen::Dynamic> f(SE2<> u, double dt, SE2<Eigen::Dynamic> state) override;
        MatrixCov MakePhi(const SE2<>& u, double dt, const SE2<Eigen::Dynamic>& state, ERROR error) override;
        
        void setQ(Eigen::Vector3d q) { this->Q_ = q.asDiagonal(); }
        void setQ(Eigen::Matrix3d q) { this->Q_ = q; }
        void setQ(double q) { this->Q_ = q*Eigen::Matrix3d::Identity(); }
};


}

#endif // ODOMETRY_PROCESS_DYNAMIC