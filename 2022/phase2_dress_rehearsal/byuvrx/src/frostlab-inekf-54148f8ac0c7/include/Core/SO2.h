#ifndef CLASS_SO2
#define CLASS_SO2

#include <Eigen/Core>
#include "LieGroup.h"

namespace InEKF {

template<int A=0>
class SO2 : public LieGroup<SO2<A>,calcStateDim(2,0,A),2,A>{
    public:
        static constexpr int rotSize = 2;
        static constexpr int N = calcStateDim(rotSize,0,A);
        static constexpr int M = calcStateMtxSize(rotSize,0);

        using typename LieGroup<SO2<A>,N,M,A>::TangentVector;
        using typename LieGroup<SO2<A>,N,M,A>::MatrixCov;
        using typename LieGroup<SO2<A>,N,M,A>::MatrixState;
        using typename LieGroup<SO2<A>,N,M,A>::VectorAug;

    private:
        // dummies to help with dynamic initialization
        static constexpr int a = A == Eigen::Dynamic ? 0 : A;
        static constexpr int c = A == Eigen::Dynamic ? 1 : N;

        using LieGroup<SO2<A>,N,M,A>::Cov_;
        using LieGroup<SO2<A>,N,M,A>::State_;
        using LieGroup<SO2<A>,N,M,A>::Aug_;
        using LieGroup<SO2<A>,N,M,A>::isUncertain;
        
        void verifySize();

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        // Constructors
        SO2(const MatrixState& State=MatrixState::Identity(), 
            const MatrixCov& Cov=MatrixCov::Zero(c,c),
            const VectorAug& Aug=VectorAug::Zero(a)) 
                : LieGroup<SO2<A>,N,M,A>(State, Cov, Aug) { verifySize(); }

        SO2(const SO2& State) : SO2(State(), State.Cov(), State.Aug()) {}

        SO2(double theta, 
            const MatrixCov& Cov=MatrixCov::Zero(c,c),
            const VectorAug& Aug=VectorAug::Zero(a)) 
                : LieGroup<SO2<A>,N,M,A>(Cov, Aug) {
            State_ << cos(theta), -sin(theta),
                    sin(theta), cos(theta);
            verifySize();
        }

        SO2(const TangentVector& xi, const MatrixCov& Cov=MatrixCov::Zero(c,c))
            : SO2(xi(0), Cov, xi.tail(xi.size()-1)) {}

        ~SO2() {}

        // Getters
        SO2<> R() const { return SO2<>(State_); }

        void addAug(double x, double sigma=1);

        // Self operations
        SO2<A> inverse() const;
        using LieGroup<SO2<A>,N,M,A>::Ad;
        using LieGroup<SO2<A>,N,M,A>::log;

        // Group action
        SO2<A> operator*(const SO2<A>& rhs) const;

        // Static Operators
        static MatrixState Wedge(const TangentVector& xi);
        static SO2 Exp(const TangentVector& xi);
        static TangentVector Log(const SO2& g);
        static MatrixCov Ad(const SO2& g);

};

}

#include "SO2.tpp"

#endif // CLASS_SO2