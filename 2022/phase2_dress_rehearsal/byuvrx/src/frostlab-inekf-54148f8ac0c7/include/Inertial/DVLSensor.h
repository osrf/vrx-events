#ifndef DVLSENSOR
#define DVLSENSOR

#include <Eigen/Core>
#include <InEKF/Core>

namespace InEKF {

class DVLSensor : public MeasureModel<SE3<2,6>> {
    
    public:
        typedef typename MeasureModel<SE3<2,6>>::MatrixS MatrixS;
        typedef typename MeasureModel<SE3<2,6>>::MatrixH MatrixH;
        typedef typename MeasureModel<SE3<2,6>>::VectorV VectorV;
        typedef typename MeasureModel<SE3<2,6>>::VectorB VectorB;

        DVLSensor();
        DVLSensor(Eigen::Matrix3d dvlR, Eigen::Vector3d dvlT);
        DVLSensor(SO3<> dvlR, Eigen::Vector3d dvlT);
        DVLSensor(SE3<> dvlH);
        ~DVLSensor(){ }

        void setNoise(double std_dvl, double std_imu);

        VectorB processZ(const Eigen::VectorXd& z, const SE3<2,6>& state) override;

    private:
        Eigen::Matrix3d dvlT_;
        SO3<> dvlR_;
};

}

#endif // DVLSENSOR