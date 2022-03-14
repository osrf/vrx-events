#ifndef GENERIC_MEASURE
#define GENERIC_MEASURE

#include <type_traits>

#include <Eigen/Core>
#include "Core/LieGroup.h"
#include "Core/MeasureModel.h"

namespace InEKF {

template<class Group>
class GenericMeasureModel : public MeasureModel<Group> {
    
    static_assert(!std::is_same<Group, SO2<>>::value,
            "GenericMeasureModel not supported for SO2");

    public:
        typedef typename MeasureModel<Group>::MatrixS MatrixS;
        typedef typename MeasureModel<Group>::MatrixH MatrixH;
        typedef typename MeasureModel<Group>::VectorV VectorV;
        typedef typename MeasureModel<Group>::VectorB VectorB;

    protected:
        VectorB b_;

    public:      
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        GenericMeasureModel(VectorB b, const MatrixS& M, ERROR error) : b_(b) {
            if(Group::N == Eigen::Dynamic){
                throw std::range_error("Can't use GenericMeasureModel on group with dynamic columns");
            }
            if(b.head(Group::rotSize) != VectorV::Zero(Group::rotSize)){
                throw std::range_error("Non-zero b in rotation portion not supported");
            }

            this->M_ = M;
            this->error_ = error;

            this->H_ = MatrixH::Zero(Group::rotSize, Group::N);

            int rDim = Group::rotSize*(Group::rotSize - 1) / 2;
            for(int i=0; i<Group::M-Group::rotSize; i++){
                this->H_.block(0, Group::rotSize*i+rDim, Group::rotSize, Group::rotSize) = b(i+Group::rotSize)*MatrixS::Identity();
            }
            if(error == ERROR::RIGHT){
                this->H_ *= -1;
            }
        }

        VectorB processZ(const Eigen::VectorXd& z, const Group& state) override {
            if(z.rows() == Group::M){
                return z;
            }
            else if(z.rows() == Group::rotSize){
                VectorB temp = b_;
                temp.head(Group::rotSize) = z;
                return temp;
            }
            else{
                throw std::range_error("Wrong sized z");
            }
        }
};

}

#endif // GENERIC_MEASURE