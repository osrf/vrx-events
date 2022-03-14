#include "Core/SO2.h"

namespace InEKF{

// helper functions
template <int A>
void SO2<A>::verifySize() {
    // we only care if it's off when it's uncertain && dynamic
    if(isUncertain && A == Eigen::Dynamic){
        int curr_A = Aug_.rows();
        int curr_dim = Cov_.rows();
        if(calcStateDim(rotSize, 0, curr_A) != curr_dim){
            throw std::range_error("Covariance size doesn't match State N");
        }
    }
}

template <int A>
void SO2<A>::addAug(double x, double sigma){
    if(A != Eigen::Dynamic) throw std::range_error("Can't add augment, not dynamic");
    
    // Add it into state
    int curr_size = Aug_.rows();
    VectorAug V(curr_size+1);
    V.head(curr_size) = Aug_;
    V(curr_size) = x;
    Aug_ = V;

    // Add into Sigma
    if(isUncertain){
        int curr_dim = Cov_.rows();

        MatrixCov Sig = MatrixCov::Zero(curr_dim+1, curr_dim+1);
        Sig.topLeftCorner(curr_dim,curr_dim) = Cov_;

        Sig(curr_dim,curr_dim) = sigma;
        Cov_ = Sig;
    }
}

template <int A>
SO2<A> SO2<A>::inverse() const{
    MatrixState temp = State_.transpose();
    return SO2(temp);
}

template <int A>
SO2<A> SO2<A>::operator*(const SO2<A>& rhs) const{
    // Make sure they're both the same size
    if(A == Eigen::Dynamic && (*this).Aug().rows() != rhs.Aug().rows()){
        throw std::range_error("Dynamic SE2 elements have different Aug");
    }

    // Skirt around composing covariances
    MatrixCov Cov = MatrixCov::Zero(c,c);
    if(this->Uncertain() && rhs.Uncertain()){
        throw "Can't compose uncertain LieGroups";
    }
    if(this->Uncertain()) Cov = this->Cov();
    if(rhs.Uncertain()) Cov = rhs.Cov();

    // Compose state + augment
    MatrixState State = (*this)() * rhs();
    VectorAug Aug = this->Aug() + rhs.Aug();

    return SO2<A>(State, Cov, Aug);
}

template <int A>
typename SO2<A>::MatrixState SO2<A>::Wedge(const TangentVector& xi){
    MatrixState State;
    double theta = xi(0);
    State << 0, -theta,
            theta, 0;
    return State;
}

template <int A>
SO2<A> SO2<A>::Exp(const TangentVector& xi){
    double theta = xi(0);
    return SO2(theta,
                MatrixCov::Zero(c,c),
                xi.tail(xi.rows()-1));
}

template <int A>
typename SO2<A>::TangentVector SO2<A>::Log(const SO2& g){
    TangentVector xi(g.Aug().rows()+1);
    xi(0) = atan2(g()(1,0), g()(0,0));
    xi.tail(xi.rows()-1) = g.Aug();
    return xi;
}

template <int A>
typename SO2<A>::MatrixCov SO2<A>::Ad(const SO2& g){
    int curr_dim = g.Aug().rows() + 1;
    return MatrixCov::Identity(curr_dim,curr_dim);
}

}