#include "Core/SO3.h"

namespace InEKF{

// helper functions
template <int A>
void SO3<A>::verifySize() {
    // we only care if it's off when it's uncertain && dynamic
    if(isUncertain && A == Eigen::Dynamic){
        int curr_A = Aug_.rows();
        int curr_dim = Cov_.rows();
        if(calcStateDim(rotSize, 0, curr_A) != curr_dim){
            throw std::range_error("Covariance size doesn't match State N");
        }
    }
}

template<int A>
SO3<A>::SO3(double w1, double w2, double w3, const MatrixCov& Cov, const VectorAug& Aug) 
        : LieGroup<SO3<A>,N,M,A>(Cov, Aug) {
    Eigen::Vector3d w;
    w << w1, w2, w3;
    double theta = w.norm();
    MatrixState wx = SO3<>::Wedge(w);

    if(abs(theta) < .0001){
        State_ = MatrixState::Identity() + wx/2 + wx*wx/6 + wx*wx*wx/24;
    }
    else{
        State_ = MatrixState::Identity() + wx*(sin(theta)/theta) + wx*wx*((1 - cos(theta))/(theta*theta));
    }
    verifySize();
}

template <int A>
void SO3<A>::addAug(double x, double sigma){
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
SO3<A> SO3<A>::inverse() const{
    MatrixState temp = State_.transpose();
    return SO3(temp);
}

template <int A>
SO3<A> SO3<A>::operator*(const SO3<A>& rhs) const{
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

    return SO3<A>(State, Cov, Aug);
}

template <int A>
typename SO3<A>::MatrixState SO3<A>::Wedge(const TangentVector& xi){
    MatrixState State;
    State <<     0, -xi(2),  xi(1),
             xi(2),      0, -xi(0),
            -xi(1),  xi(0),      0;
    return State;
}

template <int A>
SO3<A> SO3<A>::Exp(const TangentVector& xi){
    return SO3(xi, MatrixCov::Zero(c,c));
}

template <int A>
typename SO3<A>::TangentVector SO3<A>::Log(const SO3& g){
    TangentVector xi(g.Aug().rows()+3);

    double theta = acos( (g().trace()-1)/2 );
    xi(0) = g()(2,1) - g()(1,2);
    xi(1) = g()(0,2) - g()(2,0);
    xi(2) = g()(1,0) - g()(0,1);
    if(theta != 0){
        xi.head(3) *= theta / (2*sin(theta));
    }

    xi.tail(xi.rows()-3) = g.Aug();
    return xi;
}

template <int A>
typename SO3<A>::MatrixCov SO3<A>::Ad(const SO3& g){
    int curr_dim = g.Aug().rows() + 3;
    MatrixCov adjoint = MatrixCov::Identity(curr_dim,curr_dim);
    adjoint.block(0,0,3,3) = g();
    return adjoint;
}

}