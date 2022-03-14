#ifndef BASE_LIE
#define BASE_LIE

#include <Eigen/Core>
#include <string>
#include <ostream>

namespace InEKF {

enum ERROR { LEFT, RIGHT };

constexpr int calcStateDim(int rotMtxSize, int C, int A){
    if(A == Eigen::Dynamic || C == Eigen::Dynamic){
        return Eigen::Dynamic;
    }
    else{
        return rotMtxSize*(rotMtxSize-1)/2 + rotMtxSize *C + A;
    }
}
constexpr int calcStateMtxSize(int rotMtxSize, int C){
    if (C == Eigen::Dynamic){
        return Eigen::Dynamic;
    }
    else{
        return rotMtxSize + C;
    }
}

// N = Group dimension
// M = Lie Group matrix size
// A = Augmented Euclidean state size
template  <class Class, int N, int M, int A>
class LieGroup{

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef Eigen::Matrix<double, N, 1> TangentVector;
        typedef Eigen::Matrix<double, N, N> MatrixCov;
        typedef Eigen::Matrix<double, M, M> MatrixState;
        typedef Eigen::Matrix<double, A, 1> VectorAug;

    protected:
        MatrixState State_;
        MatrixCov Cov_;
        VectorAug Aug_;
        bool isUncertain;

    public:
        LieGroup() {}
        LieGroup(MatrixState State, MatrixCov Cov, VectorAug Aug) 
            : State_(State), Cov_(Cov), Aug_(Aug), isUncertain(!Cov.isZero()) {}
        LieGroup(MatrixCov Cov, VectorAug Aug) 
            : Cov_(Cov), Aug_(Aug), isUncertain(!Cov.isZero()) {}
        LieGroup(MatrixCov Cov) 
            : Cov_(Cov), isUncertain(!Cov.isZero()) {}

        virtual ~LieGroup() {};

        // Getters
        bool Uncertain() const { return isUncertain; }
        const MatrixCov& Cov() const { return Cov_; }
        const VectorAug& Aug() const { return Aug_; }
        const MatrixState& operator()() const { return State_; }
        
        // Setters
        void setCov(const MatrixCov& Cov) { Cov_ = Cov; };
        void setAug(const VectorAug& Aug) { Aug_ = Aug; };
        void setState(const MatrixState& State) { State_ = State; }

        // helper to automatically cast things
        const Class & derived() const{
            return static_cast<const Class&>(*this);
        }

        // self operations
        Class inverse() const {
            return derived().inverse();
        }
        TangentVector log() const {
            return Class::Log(derived());
        }
        MatrixCov Ad() const{
            return Class::Ad(derived());
        }

        // Group action
        Class compose(const Class& g) const {
            return derived() * g;
        }

        // static operators
        static MatrixState Wedge(const TangentVector& xi){
            return Class::Wedge(xi);
        }
        static Class Exp(const TangentVector& xi){
            return Class::Exp(xi);
        }
        static TangentVector Log(const Class& g){
            return Class::Log(g);
        }
        static MatrixCov Ad(const Class& g){
            return Class::Ad(g);
        }

        std::string toString() const{
            std::ostringstream os;

            os << "Matrix\n" << State_;
            if(isUncertain) os << "\nSigma\n" << Cov_;
            if(A != 0) os << "\nAug\n" << Aug_;

            return os.str();
        }

};

}

template <class Class, int N, int M, int A>
std::ostream& operator<<(std::ostream& os, const InEKF::LieGroup<Class,N,M,A>& rhs)  
{
    os << "Matrix\n" << rhs();
    if(rhs.Uncertain()) os << "\nSigma\n" << rhs.Cov();
    if(A != 0) os << "\nAug\n" << rhs.Aug();
    return os;
}

#endif // BASE_LIE