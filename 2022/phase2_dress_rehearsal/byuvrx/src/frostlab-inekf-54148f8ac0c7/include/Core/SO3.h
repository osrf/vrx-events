#ifndef CLASS_SO3
#define CLASS_SO3

#include <Eigen/Core>
#include "LieGroup.h"

namespace InEKF {

template<int A=0>
class SO3 : public LieGroup<SO3<A>,calcStateDim(3,0,A),3,A>{
    public:
        static constexpr int rotSize = 3;
        static constexpr int N = calcStateDim(rotSize,0,A);
        static constexpr int M = calcStateMtxSize(rotSize,0);

        using typename LieGroup<SO3<A>,N,M,A>::TangentVector;
        using typename LieGroup<SO3<A>,N,M,A>::MatrixCov;
        using typename LieGroup<SO3<A>,N,M,A>::MatrixState;
        using typename LieGroup<SO3<A>,N,M,A>::VectorAug;

    private:
        // dummies to help with dynamic initialization
        static constexpr int a = A == Eigen::Dynamic ? 0 : A;
        static constexpr int c = A == Eigen::Dynamic ? 3 : N;

        using LieGroup<SO3<A>,N,M,A>::Cov_;
        using LieGroup<SO3<A>,N,M,A>::State_;
        using LieGroup<SO3<A>,N,M,A>::Aug_;
        using LieGroup<SO3<A>,N,M,A>::isUncertain;
        
        void verifySize();

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        // Constructors
        SO3(const MatrixState& State=MatrixState::Identity(), 
            const MatrixCov& Cov=MatrixCov::Zero(c,c),
            const VectorAug& Aug=VectorAug::Zero(a)) 
                : LieGroup<SO3<A>,N,M,A>(State, Cov, Aug) { verifySize(); }

        SO3(const SO3& State) : SO3(State(), State.Cov(), State.Aug()) {}

        SO3(double w1, double w2, double w3, 
            const MatrixCov& Cov=MatrixCov::Zero(c,c),
            const VectorAug& Aug=VectorAug::Zero(a));

        SO3(const TangentVector& xi, const MatrixCov& Cov=MatrixCov::Zero(c,c))
            : SO3(xi(0), xi(1), xi(2), Cov, xi.tail(xi.size()-3)) {}

        ~SO3() {}

        // Getters
        SO3<> R() const { return SO3<>(State_); }

        void addAug(double x, double sigma=1);

        // Self operations
        SO3<A> inverse() const;
        using LieGroup<SO3<A>,N,M,A>::Ad;
        using LieGroup<SO3<A>,N,M,A>::log;

        // Group action
        SO3<A> operator*(const SO3<A>& rhs) const;

        // Static Operators
        static MatrixState Wedge(const TangentVector& xi);
        static SO3 Exp(const TangentVector& xi);
        static TangentVector Log(const SO3& g);
        static MatrixCov Ad(const SO3& g);

};

}

#include "SO3.tpp"

#endif // CLASS_SO3