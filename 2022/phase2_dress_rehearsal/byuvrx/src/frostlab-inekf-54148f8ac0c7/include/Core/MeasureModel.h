#ifndef BASE_MEASURE
#define BASE_MEASURE

#include <Eigen/Core>
#include <Eigen/LU>
#include "Core/LieGroup.h"

namespace InEKF {

template<class Group>
class MeasureModel {
    
    public:
        typedef Eigen::Matrix<double,Group::rotSize,Group::rotSize> MatrixS;
        typedef Eigen::Matrix<double,Group::rotSize,Group::N> MatrixH;
        typedef Eigen::Matrix<double,Group::rotSize,1> VectorV;
        typedef Eigen::Matrix<double,Group::M,1> VectorB;

    protected:
        // These are all constant and should be set once
        ERROR error_;
        MatrixS M_;

        // This one can be changed each iteration in InEKF.Update, 
        // or should be set once in constructor
        MatrixH H_;

        // This is changed by InEKF based on if it's a RIGHT/LEFT filter
        // Use this in calcSInverse if you override it
        MatrixH H_error_;


    public: 
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
               
        MeasureModel() {};
        MeasureModel(MatrixH H, MatrixS M, ERROR error) {
            this->H_ = H;
            this->M_ = M;
            this->error_ = error;
        };
        MeasureModel(MatrixS M, ERROR error) {
            this->M_ = M;
            this->error_ = error;
        };

        virtual VectorB processZ(const Eigen::VectorXd& z, const Group& state) { 
            if(z.rows() == Group::M){
                return z;
            }
            else{
                throw std::range_error("Wrong sized z");
            }
        }

        virtual MatrixH makeHError(const Group& state, ERROR iekfERROR){
            if( iekfERROR != error_ ){
                if(iekfERROR == ERROR::RIGHT){
                    H_error_ = H_*Group::Ad( state.inverse()() );
                }
                else{
                    H_error_ = H_*Group::Ad( state() );
                }
            }
            else{
                H_error_ = H_;
            }
            return H_error_;
        }

        virtual VectorV calcV(const VectorB& z, const Group& state){
            // calculate V
            VectorV V;
            if(error_ == ERROR::RIGHT){
                V.noalias() = state().block(0,0,Group::rotSize,state().cols()) * z;
            }
            else{
                V.noalias() = state.inverse()().block(0,0,Group::rotSize,state().cols()) * z;
            }
            return V;
        }

        virtual MatrixS calcSInverse(const Group& state){
            MatrixS Sinv;
            MatrixS R = state.R()();
            if(error_ == ERROR::RIGHT){
                Sinv.noalias() = ( H_error_*state.Cov()*H_error_.transpose() + R*M_*R.transpose() ).inverse();
            }
            else{
                Sinv.noalias() = ( H_error_*state.Cov()*H_error_.transpose() + R.transpose()*M_*R ).inverse();
            }
            return Sinv;
        }

        MatrixH getH() { return H_; }
        ERROR getError() { return error_; }

        void setH(MatrixH H) { H_ = H; }
};

}

#endif // BASE_MEASURE