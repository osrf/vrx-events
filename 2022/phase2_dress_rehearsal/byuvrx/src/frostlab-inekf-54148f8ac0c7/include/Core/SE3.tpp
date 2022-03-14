#include "Core/SE3.h"

namespace InEKF {

// helper functions
template <int C, int A>
void SE3<C,A>::verifySize() {
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
void SE3<C,A>::verifyTangentVector(const TangentVector& xi){
    // check to make sure everything is ok if things are dynamic
    if (C == Eigen::Dynamic && A == Eigen::Dynamic){
        throw std::range_error("Not supported for double Dynamic type");
    }
    if (C == Eigen::Dynamic && (xi.rows() - 3 - A) % 3 != 0){
        throw std::range_error("Tangent Vector size was incorrect");
    }
}

// initialize with theta, x, y
template <>
inline SE3<1,0>::SE3(double w1, double w2, double w3, double x, double y, double z, const MatrixCov& Cov) 
        : LieGroup<SE3<1,0>,N,M,0>(Cov)  {
    Eigen::Vector3d p;
    p << x, y, z;

    State_ = MatrixState::Identity();
    State_.block(0,0,3,3) = SO3<>(w1, w2, w3)();
    State_.block(0,3,3,1) = p;
}

template <int C, int A>
SE3<C,A>::SE3(const TangentVector& xi, const MatrixCov& Cov) 
        : LieGroup<SE3<C,A>,N,M,A>(Cov) {
    verifyTangentVector(xi);

    // figure out state size for dynamic purposes
    int curr_cols = C;
    int curr_A = A;
    if (C == Eigen::Dynamic){
        curr_cols = (xi.rows() - 3 - A) / rotSize;
    }
    else if(A == Eigen::Dynamic){
        curr_A  = (xi.rows() - 3 - C*3);
    }
    int curr_M = calcStateMtxSize(rotSize, curr_cols);

    // fill it up!
    State_ = MatrixState::Identity(curr_M, curr_M);
    State_.block(0,0,3,3) = SO3<>::Exp(xi.head(3))();
    for(int i=0;i<curr_cols;i++){
        State_.block(0,3+i,3,1) = xi.segment(3*i+3,3);
    }
    Aug_ = xi.tail(curr_A);
    verifySize();
}

template <int C, int A>
SE3<C,A>::SE3(const SO3<> R, const Eigen::Matrix<double,small_xi,1>& xi, const MatrixCov& Cov)
        : LieGroup<SE3<C,A>,N,M,A>(Cov) {
    TangentVector xi_copy = TangentVector::Zero(xi.rows()+3);
    xi_copy.tail(xi.rows()) = xi;
    verifyTangentVector(xi_copy);

    // figure out state size for dynamic purposes
    int curr_cols = C;
    int curr_A = A;
    if (C == Eigen::Dynamic){
        curr_cols = (xi.rows() - A) / rotSize;
    }
    else if(A == Eigen::Dynamic){
        curr_A  = (xi.rows() - C*3);
    }
    int curr_M = calcStateMtxSize(rotSize, curr_cols);

    // fill it up!
    State_ = MatrixState::Identity(curr_M, curr_M);
    State_.block(0,0,3,3) = R();
    for(int i=0;i<curr_cols;i++){
        State_.block(0,3+i,3,1) = xi.segment(3*i,3);
    }
    Aug_ = xi.tail(curr_A);
    verifySize();
}

template <int C, int A>
void SE3<C,A>::addCol(const Eigen::Vector3d& x, const Eigen::Matrix3d& sigma){
    if (C != Eigen::Dynamic) throw std::range_error("Can't add columns, not dynamic");
    
    // Add it into state
    int curr_size = State_.rows();
    MatrixState S = MatrixState::Identity(curr_size+1, curr_size+1);
    S.block(0, 0, curr_size, curr_size) = State_;
    S.block(0,curr_size,3,1) = x;
    State_ = S;

    // Add into Sigma
    if(isUncertain){
        int curr_A = Aug_.rows();
        int curr_dim = Cov_.rows();
        int mtx_dim = curr_dim - curr_A;

        MatrixCov Sig = MatrixCov::Zero(curr_dim+3, curr_dim+3);
        Sig.topLeftCorner(mtx_dim,mtx_dim) = Cov_.topLeftCorner(mtx_dim,mtx_dim);
        Sig.bottomRightCorner(curr_A,curr_A) = Cov_.bottomRightCorner(curr_A,curr_A);
        Sig.bottomLeftCorner(curr_A, mtx_dim) = Cov_.bottomLeftCorner(curr_A, mtx_dim);
        Sig.topRightCorner(mtx_dim, curr_A) = Cov_.topRightCorner(mtx_dim, curr_A);

        Sig.block(mtx_dim, mtx_dim, 3, 3) = sigma;
        Cov_ = Sig;
    }
}

template <int C, int A>
void SE3<C,A>::addAug(double x, double sigma){
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
SE3<C,A> SE3<C, A>::Exp(const TangentVector& xi){
    verifyTangentVector(xi);
    double theta = xi.head(3).norm();

    // Find V
    Eigen::Matrix3d V;
    Eigen::Matrix3d wx = SO3<>::Wedge(xi.head(3));
    if(abs(theta) < .0001){
        V = Eigen::Matrix3d::Identity() + wx/2 + wx*wx/6 + wx*wx*wx/24;
    }
    else{
        double Ai = sin(theta)/theta;
        double Bi = (1 - cos(theta)) / (theta*theta);
        double Ci = (1 - Ai) / (theta*theta);
        V = Eigen::Matrix3d::Identity() + wx*Bi + wx*wx*Ci;
    }

    // figure out state size for dynamic purposes
    int curr_cols = C;
    int curr_A = A;
    if (C == Eigen::Dynamic){
        curr_cols = (xi.rows() - 3 - A) / rotSize;
    }
    else if(A == Eigen::Dynamic){
        curr_A  = (xi.rows() - 3 - C*3);
    }
    int curr_M = calcStateMtxSize(rotSize, curr_cols);


    MatrixState X = MatrixState::Identity(curr_M,curr_M);
    X.block(0,0,3,3) = SO3<>::Exp(xi.head(3))();
    for(int i=0;i<curr_cols;i++){
        X.block(0,3+i,3,1) = V*xi.segment(3*i+3,3);
    }
    return SE3(X, MatrixCov::Zero(c,c), xi.tail(curr_A));
}

template <int C, int A>
SE3<C,A> SE3<C,A>::inverse() const{
    int curr_cols = State_.cols() - rotSize;
    int curr_size  = State_.cols();

    MatrixState S = MatrixState::Identity(curr_size,curr_size);
    Eigen::Matrix3d RT = this->R().inverse()();
    S.block(0,0,3,3) = RT;
    for(int i=0;i<curr_cols;i++){
        S.block(0,3+i,3,1) = -1 * RT * (*this)[i];
    }
    return SE3(S);
}

template <int C, int A>
SE3<C,A> SE3<C,A>::operator*(const SE3& rhs) const{
    // Make sure they're both the same size
    if (C == Eigen::Dynamic && (*this)().cols() != rhs().cols()){
        throw std::range_error("Dynamic SE3 elements have different C");
    }
    if(A == Eigen::Dynamic && (*this).Aug().rows() != rhs.Aug().rows()){
        throw std::range_error("Dynamic SE3 elements have different Aug");
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
        State.block(0,3+i,rotSize,1) = thisR*rhs[i] + (*this)[i];
    }

    return SE3(State, Cov, Aug);
}

template <int C, int A>
typename SE3<C,A>::MatrixState SE3<C,A>::Wedge(const TangentVector& xi){
    verifyTangentVector(xi);

    // figure out state size for dynamic purposes
    int curr_cols = C;
    int curr_A = A;
    if (C == Eigen::Dynamic){
        curr_cols = (xi.rows() - 3 - A) / rotSize;
    }
    else if(A == Eigen::Dynamic){
        curr_A  = (xi.rows() - 3 - C*3);
    }
    int curr_M = calcStateMtxSize(rotSize, curr_cols);

    // Fill it in
    MatrixState X = MatrixState::Zero(curr_M, curr_M);
    X.block(0,0,3,3) = SO3<>::Wedge(xi.head(3));
    for(int i=0;i<curr_cols;i++){
        X.block(0,3+i,3,1) = xi.segment(3*i+3,3);
    }
    return X;
}

template <int C, int A>
typename SE3<C,A>::TangentVector SE3<C,A>::Log(const SE3& g){
    // figure out state size for dynamic purposes
    int curr_cols = g().cols() - rotSize;
    int curr_A = g.Aug().rows();
    int curr_dim = calcStateDim(rotSize, curr_cols, curr_A);

    // Find angles
    double theta = acos( (g.R()().trace()-1)/2 );
    Eigen::Vector3d w = SO3<>::Log(g.R());

    // Find V inverse
    Eigen::Matrix3d V;
    Eigen::Matrix3d wx = SO3<>::Wedge(w);
    if(abs(theta) < .0001){
        V = Eigen::Matrix3d::Identity() + wx/2 + wx*wx/6 + wx*wx*wx/24;
    }
    else{
        double Ai = sin(theta)/theta;
        double Bi = (1 - cos(theta)) / (theta*theta);
        double Ci = (1 - Ai) / (theta*theta);
        V = Eigen::Matrix3d::Identity() + wx*Bi + wx*wx*Ci;
    }
    Eigen::Matrix3d V_inv = V.inverse();

    // Put it all together
    TangentVector xi = TangentVector::Zero(curr_dim);
    xi.head(3) = w;
    for(int i=0;i<curr_cols;i++){
        xi.segment(3+3*i,3) = V_inv*g[i];
    }
    xi.tail(curr_A) = g.Aug();

    return xi;
}

template <int C, int A>
typename SE3<C,A>::MatrixCov SE3<C,A>::Ad(const SE3& g){
    // figure out state size for dynamic purposes
    int curr_cols = g().cols() - rotSize;
    int curr_A = g.Aug().rows();
    int curr_dim = calcStateDim(rotSize, curr_cols, curr_A);

    MatrixCov Ad_X = MatrixCov::Identity(curr_dim, curr_dim);
    Ad_X.block(0,0,3,3) = g.R()();
    for(int i=0;i<curr_cols;i++){
        Ad_X.block(3+3*i, 3+3*i, 3, 3) = g.R()();
        Ad_X.block(3+3*i,     0, 3, 3) = SO3<>::Wedge(g[i])*g.R()();
    }
    return Ad_X;
}

}