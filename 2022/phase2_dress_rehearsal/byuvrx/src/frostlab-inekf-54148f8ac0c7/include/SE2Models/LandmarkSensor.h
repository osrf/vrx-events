#ifndef LANDMARKSENSOR
#define LANDMARKSENSOR

#include <Eigen/Core>
#include <InEKF/Core>

namespace InEKF {

class LandmarkSensor : public MeasureModel<SE2<Eigen::Dynamic>> {

    public:
        typedef typename MeasureModel<SE2<Eigen::Dynamic>>::MatrixS MatrixS;
        typedef typename MeasureModel<SE2<Eigen::Dynamic>>::MatrixH MatrixH;
        typedef typename MeasureModel<SE2<Eigen::Dynamic>>::VectorV VectorV;
        typedef typename MeasureModel<SE2<Eigen::Dynamic>>::VectorB VectorB;

    private:
        Eigen::VectorXd b_;
        MatrixS M_rb;
        Eigen::Matrix<double,2,4> HSmall;
        int lmIdx;
        ERROR filterError;
        
    public:
        LandmarkSensor(double std_r, double std_b);
        ~LandmarkSensor(){ }
        // Used to prep H
        void sawLandmark(int idx, const SE2<Eigen::Dynamic>& state);
        // used for data assocation, to reduce calls to C++
        double calcMahDist(const Eigen::VectorXd& z, const SE2<Eigen::Dynamic>& state);

        VectorB processZ(const Eigen::VectorXd& z, const SE2<Eigen::Dynamic>& state) override;
        MatrixS calcSInverse(const SE2<Eigen::Dynamic>& state) override;
        MatrixH makeHError(const SE2<Eigen::Dynamic>& state, ERROR iekfERROR) override {
            filterError = iekfERROR;
            return MeasureModel<SE2<Eigen::Dynamic>>::makeHError(state, iekfERROR);
        }

};

}

#endif // LANDMARKSENSOR