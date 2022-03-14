#ifndef CLASS_SE2
#define CLASS_SE2

#include <Eigen/Core>
#include "LieGroup.h"
#include "SO2.h"

namespace InEKF {

template <int C=1, int A=0>
class SE2 : public LieGroup<SE2<C,A>,calcStateDim(2,C,A),calcStateMtxSize(2,C),A>{
    public:
        static constexpr int rotSize = 2;
        static constexpr int N = calcStateDim(rotSize,C,A);
        static constexpr int M = calcStateMtxSize(rotSize,C);

        typedef typename LieGroup<SE2<C,A>,N,M,A>::TangentVector TangentVector;
        typedef typename LieGroup<SE2<C,A>,N,M,A>::MatrixCov MatrixCov;
        typedef typename LieGroup<SE2<C,A>,N,M,A>::MatrixState MatrixState;
        typedef typename LieGroup<SE2<C,A>,N,M,A>::VectorAug VectorAug;

    private:
        // dummies to help with dynamic initialization
        static constexpr int a = A == Eigen::Dynamic ? 0 : A;
        static constexpr int c = N == Eigen::Dynamic ? 3 : N;
        static constexpr int m = M == Eigen::Dynamic ? 3 : M;

        using LieGroup<SE2<C,A>,N,M,A>::Cov_;
        using LieGroup<SE2<C,A>,N,M,A>::State_;
        using LieGroup<SE2<C,A>,N,M,A>::Aug_;
        using LieGroup<SE2<C,A>,N,M,A>::isUncertain;

        void verifySize();
        static void verifyTangentVector(const TangentVector& xi);

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        // Constructors
        SE2(const MatrixState& State=MatrixState::Identity(m,m), 
            const MatrixCov& Cov=MatrixCov::Zero(c,c),
            const VectorAug& Aug=VectorAug::Zero(a,1))
                : LieGroup<SE2<C,A>,N,M,A>(State, Cov, Aug) { verifySize(); }
        // TODO: Turn this off for dynamic sizes
        SE2(bool uncertain) : SE2() {
            Cov_ = MatrixCov::Identity(c,c);
            isUncertain = uncertain;
        }
        SE2(const SE2& State) : SE2(State(), State.Cov(), State.Aug()) {}
        SE2(const TangentVector& xi,
            const MatrixCov& Cov=MatrixCov::Zero(c,c));
        SE2(double theta, double x, double y,
            const MatrixCov& Cov=MatrixCov::Zero(c,c)) { 
                throw std::invalid_argument("Can't use this constructor with those templates");
        }
        ~SE2() {}

        // Getters
        SO2<> R() const { 
            Eigen::Matrix2d R = State_.block(0,0,2,2);
            return SO2<>(R); 
        }
        Eigen::Vector2d operator[](int idx) const { 
            int curr_cols = State_.cols() - rotSize;
            if(idx >= curr_cols){
                throw std::out_of_range("Sliced out of range");
            }
            return State_.block(0,2+idx,2,1); 
        }

        void addCol(const Eigen::Vector2d& x, const Eigen::Matrix2d& sigma=Eigen::Matrix2d::Identity());
        void addAug(double x, double sigma=1);

        // Self operations
        SE2 inverse() const;
        using LieGroup<SE2<C,A>,N,M,A>::Ad;
        using LieGroup<SE2<C,A>,N,M,A>::log;

        // Group action
        SE2 operator*(const SE2& rhs) const;

        // Static Operators
        static MatrixState Wedge(const TangentVector& xi);
        static SE2 Exp(const TangentVector& xi);
        static TangentVector Log(const SE2& g);
        static MatrixCov Ad(const SE2& g);

};

}

#include "SE2.tpp"

#endif // CLASS_SE2