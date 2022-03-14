#ifndef GPSSENSOR
#define GPSSENSOR

#include <Eigen/Core>
#include <InEKF/Core>

namespace InEKF {

class GPSSensor : public MeasureModel<SE2<Eigen::Dynamic>> {
    public:
        typedef typename MeasureModel<SE2<Eigen::Dynamic>>::MatrixS MatrixS;
        typedef typename MeasureModel<SE2<Eigen::Dynamic>>::MatrixH MatrixH;
        typedef typename MeasureModel<SE2<Eigen::Dynamic>>::VectorV VectorV;
        typedef typename MeasureModel<SE2<Eigen::Dynamic>>::VectorB VectorB;

        GPSSensor(double std=1);
        ~GPSSensor(){ }
        // Used to prep H
        VectorB processZ(const Eigen::VectorXd& z, const SE2<Eigen::Dynamic>& state) override;
};

}

#endif // GPSSENSOR