#include "Core/SE2.h"

namespace InEKF {

// helper functions
template <int C, int A>
void SE2<C,A>::verifySize() {
    if(State_.rows() != State_.cols()){
        throw std::invalid_argument("Matrix passed is not square");
    }
    // we only care if it's off when it's uncertain && dynamic
    if(isUncertain &&  (C == Eigen::Dynamic || A == Eigen::Dynamic)){
        if(Cov_.rows() != Cov_.cols()){
            throw std::invalid_argument("Covariance passed is not square");
        }
        int curr_A = Aug_.rows();
        int curr_cols = State_.rows() - rotSize;
        int curr_dim = Cov_.rows();
        if(calcStateDim(rotSize, curr_cols, curr_A) != curr_dim){
            throw std::range_error("Covariance size doesn't match State Dimension");
        }
    }
}

template <int C, int A>
void SE2<C,A>::verifyTangentVector(const TangentVector& xi){
    // check to make sure everything is ok if things are dynamic
    if (C == Eigen::Dynamic && A == Eigen::Dynamic){
        throw std::range_error("Not supported for double Dynamic type");
    }
    if (C == Eigen::Dynamic && (xi.rows() - 1 - A) % 2 != 0){
        throw std::range_error("Tangent Vector size was incorrect");
    }
}

// initialize with theta, x, y
template <>
inline SE2<1,0>::SE2(double theta, double x, double y, const MatrixCov& Cov) 
        : LieGroup<SE2<1,0>,N,M,0>(Cov)  {
    State_ << cos(theta), -sin(theta), x,
                sin(theta),  cos(theta), y,
                0, 0, 1;
}

template <int C, int A>
SE2<C,A>::SE2(const TangentVector& xi, const MatrixCov& Cov) 
        : LieGroup<SE2<C,A>,N,M,A>(Cov) {
    verifyTangentVector(xi);

    // figure out state size for dynamic purposes
    int curr_cols = C;
    int curr_A = A;
    if (C == Eigen::Dynamic){
        curr_cols = (xi.rows() - 1 - A) / rotSize;
    }
    else if(A == Eigen::Dynamic){
        curr_A  = (xi.rows() - 1 - C*2);
    }
    int curr_M = calcStateMtxSize(rotSize, curr_cols);

    // fill it up!
    State_ = MatrixState::Identity(curr_M, curr_M);
    State_.block(0,0,2,2) = SO2<>(xi(0))();
    for(int i=0;i<curr_cols;i++){
        State_.block(0,2+i,2,1) = xi.segment(2*i+1,2);
    }
    Aug_ = xi.tail(curr_A);
    verifySize();
}

template <int C, int A>
void SE2<C,A>::addCol(const Eigen::Vector2d& x, const Eigen::Matrix2d& sigma){
    if (C != Eigen::Dynamic) throw std::range_error("Can't add columns, not dynamic");
    
    // Add it into state
    int curr_size = State_.rows();
    MatrixState S = MatrixState::Identity(curr_size+1, curr_size+1);
    S.block(0, 0, curr_size, curr_size) = State_;
    S.block(0,curr_size,2,1) = x;
    State_ = S;

    // Add into Sigma
    if(isUncertain){
        int curr_A = Aug_.rows();
        int curr_dim = Cov_.rows();
        int mtx_dim = curr_dim - curr_A;

        MatrixCov Sig = MatrixCov::Zero(curr_dim+2, curr_dim+2);
        Sig.topLeftCorner(mtx_dim,mtx_dim) = Cov_.topLeftCorner(mtx_dim,mtx_dim);
        Sig.bottomRightCorner(curr_A,curr_A) = Cov_.bottomRightCorner(curr_A,curr_A);
        Sig.bottomLeftCorner(curr_A, mtx_dim) = Cov_.bottomLeftCorner(curr_A, mtx_dim);
        Sig.topRightCorner(mtx_dim, curr_A) = Cov_.topRightCorner(mtx_dim, curr_A);

        Sig.block(mtx_dim, mtx_dim, 2, 2) = sigma;
        Cov_ = Sig;
    }
}

template <int C, int A>
void SE2<C,A>::addAug(double x, double sigma){
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

template <int C, int A>
SE2<C,A> SE2<C, A>::Exp(const TangentVector& xi){
    verifyTangentVector(xi);
    double theta = xi(0);

    // Find V
    Eigen::Matrix2d V;
    if(abs(theta) < .0001){
        Eigen::Matrix2d wx = SO2<>::Wedge(xi.segment(0,1));
        V = Eigen::Matrix2d::Identity() + wx/2 + wx*wx/6 + wx*wx*wx/24;
    }
    else{
        V(0,0) = sin(theta);
        V(1,1) = sin(theta);
        V(1,0) = 1 - cos(theta);
        V(0,1) = cos(theta) - 1;
        V /= theta;
    }

    // figure out state size for dynamic purposes
    int curr_cols = C;
    int curr_A = A;
    if (C == Eigen::Dynamic){
        curr_cols = (xi.rows() - 1 - A) / rotSize;
    }
    else if(A == Eigen::Dynamic){
        curr_A  = (xi.rows() - 1 - C*2);
    }
    int curr_M = calcStateMtxSize(rotSize, curr_cols);

    MatrixState X = MatrixState::Identity(curr_M,curr_M);
    X.block(0,0,2,2) = SO2<>::Exp(xi.segment(0,1))();
    for(int i=0;i<curr_cols;i++){
        X.block(0,2+i,2,1) = V*xi.segment(2*i+1,2);
    }
    return SE2(X, MatrixCov::Zero(c,c), xi.tail(curr_A));
}

template <int C, int A>
SE2<C,A> SE2<C,A>::inverse() const{
    int curr_cols = State_.cols() - rotSize;
    int curr_size  = State_.cols();

    MatrixState S = MatrixState::Identity(curr_size,curr_size);
    Eigen::Matrix2d RT = this->R().inverse()();
    S.block(0,0,2,2) = RT;
    for(int i=0;i<curr_cols;i++){
        S.block(0,2+i,2,1) = -1 * RT * (*this)[i];
    }
    return SE2(S);
}

template <int C, int A>
SE2<C,A> SE2<C,A>::operator*(const SE2& rhs) const{
    // Make sure they're both the same size
    if (C == Eigen::Dynamic && (*this)().cols() != rhs().cols()){
        throw std::range_error("Dynamic SE2 elements have different C");
    }
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

    // Compose Augment
    VectorAug Aug = this->Aug() + rhs.Aug();

    // Smart multiply state matrix
    int curr_cols = (*this)().cols() - rotSize;
    Eigen::Matrix<double,rotSize,rotSize> thisR = this->R()(); 
    MatrixState State = MatrixState::Identity((*this)().cols(), (*this)().cols());

    State.block(0,0,rotSize,rotSize) = thisR*rhs().block(0,0,rotSize,rotSize);
    for(int i=0;i<curr_cols;i++){
        State.block(0,2+i,rotSize,1) = thisR*rhs[i] + (*this)[i];
    }
    return SE2(State, Cov, Aug);
}

template <int C, int A>
typename SE2<C,A>::MatrixState SE2<C,A>::Wedge(const TangentVector& xi){
    verifyTangentVector(xi);

    // figure out state size for dynamic purposes
    int curr_cols = C;
    int curr_A = A;
    if (C == Eigen::Dynamic){
        curr_cols = (xi.rows() - 1 - A) / rotSize;
    }
    else if(A == Eigen::Dynamic){
        curr_A  = (xi.rows() - 1 - C*2);
    }
    int curr_M = calcStateMtxSize(rotSize, curr_cols);

    // Fill it in
    MatrixState X = MatrixState::Zero(curr_M, curr_M);
    X.block(0,0,2,2) = SO2<>::Wedge(xi.segment(0,1));
    for(int i=0;i<curr_cols;i++){
        X.block(0,2+i,2,1) = xi.segment(2*i+1,2);
    }
    return X;
}

template <int C, int A>
typename SE2<C,A>::TangentVector SE2<C,A>::Log(const SE2& g){
    // figure out state size for dynamic purposes
    int curr_cols = g().cols() - rotSize;
    int curr_A = g.Aug().rows();
    int curr_dim = calcStateDim(rotSize, curr_cols, curr_A);

    double theta = SO2<>::Log(g.R())(0);
    double a, b, scale;
    if(abs(theta) < .0001){
        a = 1;
        b = 0;
        scale = 1;
    }
    else{
        a = sin(theta)/theta;
        b = (1-cos(theta))/theta;
        scale = 1 / (a*a + b*b);
    }

    Eigen::Matrix2d V_inv;
    V_inv << a, b, -b, a;
    V_inv *= scale;

    TangentVector xi = TangentVector::Zero(curr_dim);
    xi(0) = theta;
    for(int i=0;i<curr_cols;i++){
        xi.segment(1+2*i,2) = V_inv*g[i];
    }
    xi.tail(curr_A) = g.Aug();
    return xi;
}

template <int C, int A>
typename SE2<C,A>::MatrixCov SE2<C,A>::Ad(const SE2& g){
    // figure out state size for dynamic purposes
    int curr_cols = g().cols() - rotSize;
    int curr_A = g.Aug().rows();
    int curr_dim = calcStateDim(rotSize, curr_cols, curr_A);

    MatrixCov Ad_X = MatrixCov::Identity(curr_dim, curr_dim);
    for(int i=0;i<curr_cols;i++){
        Ad_X.block(1+2*i,1+2*i,2,2) = g.R()();
        Ad_X(1+2*i,0) = g[i][1];
        Ad_X(2+2*i,0) = -g[i][0];
    }
    return Ad_X;
}

}