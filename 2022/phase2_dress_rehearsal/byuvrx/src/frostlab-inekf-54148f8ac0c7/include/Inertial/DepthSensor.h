#ifndef DEPTHSENSOR
#define DEPTHSENSOR

#include <Eigen/Core>
#include <InEKF/Core>

namespace InEKF {

class DepthSensor : public MeasureModel<SE3<2,6>> {
    
    public:
        typedef typename MeasureModel<SE3<2,6>>::MatrixS MatrixS;
        typedef typename MeasureModel<SE3<2,6>>::MatrixH MatrixH;
        typedef typename MeasureModel<SE3<2,6>>::VectorV VectorV;
        typedef typename MeasureModel<SE3<2,6>>::VectorB VectorB;

        DepthSensor(double std=1);
        ~DepthSensor(){ }
        VectorB processZ(const Eigen::VectorXd& z, const SE3<2,6>& state) override;
        MatrixS calcSInverse(const SE3<2,6>& state) override;
        void setNoise(double std);
};

}

#endif // DEPTHSENSOR