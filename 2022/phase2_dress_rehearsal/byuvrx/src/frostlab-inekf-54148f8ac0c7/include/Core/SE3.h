#ifndef CLASS_SE3
#define CLASS_SE3

#include <Eigen/Core>
#include "LieGroup.h"
#include "SO3.h"

namespace InEKF {

template <int C=1, int A=0>
class SE3 : public LieGroup<SE3<C,A>,calcStateDim(3,C,A),calcStateMtxSize(3,C),A>{
    public:
        static constexpr int rotSize = 3;
        static constexpr int N = calcStateDim(rotSize,C,A);
        static constexpr int M = calcStateMtxSize(rotSize,C);

        typedef typename LieGroup<SE3<C,A>,N,M,A>::TangentVector TangentVector;
        typedef typename LieGroup<SE3<C,A>,N,M,A>::MatrixCov MatrixCov;
        typedef typename LieGroup<SE3<C,A>,N,M,A>::MatrixState MatrixState;
        typedef typename LieGroup<SE3<C,A>,N,M,A>::VectorAug VectorAug;

    private:
        // dummies to help with dynamic initialization
        static constexpr int a = A == Eigen::Dynamic ? 0 : A;
        static constexpr int c = N == Eigen::Dynamic ? 6 : N;
        static constexpr int m = M == Eigen::Dynamic ? 4 : M;

        static constexpr int small_xi = N == Eigen::Dynamic ? Eigen::Dynamic : N-3;

        using LieGroup<SE3<C,A>,N,M,A>::Cov_;
        using LieGroup<SE3<C,A>,N,M,A>::State_;
        using LieGroup<SE3<C,A>,N,M,A>::Aug_;
        using LieGroup<SE3<C,A>,N,M,A>::isUncertain;

        void verifySize();
        static void verifyTangentVector(const TangentVector& xi);

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        // Constructors
        SE3(const MatrixState& State=MatrixState::Identity(m,m), 
            const MatrixCov& Cov=MatrixCov::Zero(c,c),
            const VectorAug& Aug=VectorAug::Zero(a,1))
                : LieGroup<SE3<C,A>,N,M,A>(State, Cov, Aug) { verifySize(); }
        // TODO: Turn this off for dynamic sizes
        SE3(bool uncertain) : SE3() {
            Cov_ = MatrixCov::Identity(c,c);
            isUncertain = uncertain;
        }
        SE3(const SE3& State) : SE3(State(), State.Cov(), State.Aug()) {}
        SE3(const TangentVector& xi,
            const MatrixCov& Cov=MatrixCov::Zero(c,c));
        // TODO the -3 here is going to wreck havoc on dynamic types
        // Fix in python side too
        SE3(const SO3<> R, const Eigen::Matrix<double,small_xi,1>& xi,
            const MatrixCov& Cov=MatrixCov::Zero(c,c));
        SE3(double w1, double w2, double w3, double x, double y, double z,
            const MatrixCov& Cov=MatrixCov::Zero(c,c)){ 
                throw std::invalid_argument("Can't use this constructor with those templates");
        }
        ~SE3() {}

        // Getters
        SO3<> R() const { 
            Eigen::Matrix3d R = State_.block(0,0,3,3);
            return SO3<>(R); 
        }
        Eigen::Vector3d operator[](int idx) const {
            int curr_cols = State_.cols() - rotSize;
            if(idx >= curr_cols){
                throw std::out_of_range("Sliced out of range");
            }
            return State_.block(0,3+idx,3,1); 
        }

        void addCol(const Eigen::Vector3d& x, const Eigen::Matrix3d& sigma=Eigen::Matrix3d::Identity());
        void addAug(double x, double sigma=1);

        // Self operations
        SE3 inverse() const;
        using LieGroup<SE3<C,A>,N,M,A>::Ad;
        using LieGroup<SE3<C,A>,N,M,A>::log;

        // Group action
        SE3 operator*(const SE3& rhs) const;

        // Static Operators
        static MatrixState Wedge(const TangentVector& xi);
        static SE3 Exp(const TangentVector& xi);
        static TangentVector Log(const SE3& g);
        static MatrixCov Ad(const SE3& g);

};

}

#include "SE3.tpp"

#endif // CLASS_SE3