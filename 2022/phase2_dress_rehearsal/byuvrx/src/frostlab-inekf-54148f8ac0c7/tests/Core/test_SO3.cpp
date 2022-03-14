#include "gtest/gtest.h"
#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>
#include <InEKF/Core>

#define EXPECT_MATRICES_EQ(M_actual, M_expected) \
  EXPECT_TRUE(M_actual.isApprox(M_expected, 1e-6)) << "  Actual:\n" << M_actual << "\nExpected:\n" << M_expected

#define PI    3.14159265358
#define SQRT2 1.41421356237

namespace Eigen{
    typedef Matrix<double,5,5> Matrix5d;
}

TEST(SO3, BaseConstructor1){
    Eigen::Matrix3d mtx = Eigen::Matrix3d::Identity();
    Eigen::Matrix5d sigma = Eigen::Matrix5d::Identity();
    Eigen::Vector2d A = Eigen::Vector2d::Ones();

    InEKF::SO3<2> x(mtx, sigma, A);

    EXPECT_MATRICES_EQ(mtx, x());
    EXPECT_MATRICES_EQ(sigma, x.Cov());
    EXPECT_MATRICES_EQ(A, x.Aug());
    EXPECT_TRUE(x.Uncertain());
}

TEST(SO3, BaseConstructor2){
    Eigen::Matrix3d mtx = Eigen::Matrix3d::Identity();
    Eigen::Matrix5d sigma = Eigen::Matrix5d::Identity();
    Eigen::VectorXd A = Eigen::Vector2d::Ones();

    InEKF::SO3<Eigen::Dynamic> x(mtx, sigma, A);

    // make sure it gets mad if we pass the wrong sigma
    EXPECT_THROW(InEKF::SO3<Eigen::Dynamic> x(mtx, sigma);, std::range_error);
    // Or doesn't if we pass the right one
    EXPECT_NO_THROW(  (InEKF::SO3<Eigen::Dynamic>(mtx, sigma, A)) );
    // default also works
    EXPECT_NO_THROW( (InEKF::SO3<Eigen::Dynamic>()) );
}

TEST(SO3, ThetaConstructor){
    InEKF::SO3<> x(PI/4, 0, 0);
    Eigen::Matrix3d r;
    r << 1, 0, 0,
        0, 1/SQRT2, -1/SQRT2, 
        0, 1/SQRT2, 1/SQRT2;

    EXPECT_MATRICES_EQ(x(), r);
}

TEST(SO3, TangentConstructor){
    Eigen::VectorXd x(5);
    x << 0, 0, 0, 4, 5;

    InEKF::SO3<Eigen::Dynamic> state(x);
    EXPECT_MATRICES_EQ(state.Aug(), x.tail(2));
    EXPECT_MATRICES_EQ(state(), Eigen::Matrix3d::Identity());
}

TEST(SO3, AddAug){
    InEKF::SO3<Eigen::Dynamic> x;
    EXPECT_EQ(x.Aug().rows(), 0);

    x.addAug(2);

    EXPECT_EQ(x.Aug()(0), 2);

    // TODO: Test adding to Cov

    InEKF::SE2<> y;
    EXPECT_THROW( y.addAug(2), std::range_error);
}

TEST(SO3, Inverse){
    InEKF::SO3<> x(PI/4, PI/4, PI/4);

    EXPECT_MATRICES_EQ(x.inverse()(), x().inverse());
}

TEST(SO3, Exp){
    Eigen::VectorXd x(5);
    x << 1,2,3,4,5;

    InEKF::SO3<Eigen::Dynamic> ours = InEKF::SO3<Eigen::Dynamic>::Exp(x);
    Eigen::Matrix3d theirs = InEKF::SO3<Eigen::Dynamic>::Wedge(x).exp();

    EXPECT_MATRICES_EQ(ours(), theirs);
    EXPECT_MATRICES_EQ(ours.Aug(), x.tail(2));
}

TEST(SO3, Log){
    Eigen::VectorXd x(5);
    x << .1, .2, .3, 5, 6;
    InEKF::SO3<Eigen::Dynamic> state(x);

    Eigen::Matrix<double,5,1> xi = state.log();
    EXPECT_MATRICES_EQ(xi, x);
}

TEST(SO3, Wedge){
    Eigen::VectorXd x(5);
    x << 1, 2, 3, 4, 5;

    Eigen::Matrix3d ours = InEKF::SO3<Eigen::Dynamic>::Wedge(x); 
    Eigen::Matrix3d theirs;
    theirs << 0, -3, 2,
            3, 0, -1,
            -2, 1, 0;

    EXPECT_MATRICES_EQ(ours, theirs);
}

TEST(SO3, Adjoint){
    InEKF::SO3<1> x(1,1,1);

    Eigen::Matrix4d expected = Eigen::Matrix4d::Identity();
    expected.block<3,3>(0,0) = x();

    EXPECT_MATRICES_EQ(x.Ad(), expected);
    EXPECT_MATRICES_EQ(InEKF::SO3<1>::Ad(x), expected);
}