#include "gtest/gtest.h"
#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>
#include <InEKF/Core>

#define EXPECT_MATRICES_EQ(M_actual, M_expected) \
  EXPECT_TRUE(M_actual.isApprox(M_expected, 1e-6)) << "  Actual:\n" << M_actual << "\nExpected:\n" << M_expected

#define PI    3.14159265358
#define SQRT2 1.41421356237

TEST(SO2, BaseConstructor1){
    Eigen::Matrix2d mtx = Eigen::Matrix2d::Identity();
    Eigen::Matrix3d sigma = Eigen::Matrix3d::Identity();
    Eigen::Vector2d A = Eigen::Vector2d::Ones();

    InEKF::SO2<2> x(mtx, sigma, A);

    EXPECT_MATRICES_EQ(mtx, x());
    EXPECT_MATRICES_EQ(sigma, x.Cov());
    EXPECT_MATRICES_EQ(A, x.Aug());
    EXPECT_TRUE(x.Uncertain());
}

TEST(SO2, BaseConstructor2){
    Eigen::Matrix2d mtx = Eigen::Matrix2d::Identity();
    Eigen::Matrix3d sigma = Eigen::Matrix3d::Identity();
    Eigen::VectorXd A = Eigen::Vector2d::Ones();

    InEKF::SO2<Eigen::Dynamic> x(mtx, sigma, A);

    // make sure it gets mad if we pass the wrong sigma
    EXPECT_THROW(InEKF::SO2<Eigen::Dynamic> x(mtx, sigma);, std::range_error);
    // Or doesn't if we pass the right one
    EXPECT_NO_THROW(  (InEKF::SO2<Eigen::Dynamic>(mtx, sigma, A)) );
    // default also works
    EXPECT_NO_THROW( (InEKF::SO2<Eigen::Dynamic>()) );
}

TEST(SO2, ThetaConstructor){
    InEKF::SO2<> x(PI/4);
    Eigen::Matrix2d r;
    r << 1/SQRT2, -1/SQRT2, 1/SQRT2, 1/SQRT2;

    EXPECT_MATRICES_EQ(x(), r);
}

TEST(SO2, TangentConstructor){
    Eigen::VectorXd x(3);
    x << 0, 1, 2;

    InEKF::SO2<Eigen::Dynamic> state(x);
    EXPECT_MATRICES_EQ(state.Aug(), x.tail(2));
    EXPECT_MATRICES_EQ(state(), Eigen::Matrix2d::Identity());
}

TEST(SO2, AddAug){
    InEKF::SO2<Eigen::Dynamic> x;
    EXPECT_EQ(x.Aug().rows(), 0);

    x.addAug(2);

    EXPECT_EQ(x.Aug()(0), 2);

    // TODO: Test adding to Cov

    InEKF::SE2<> y;
    EXPECT_THROW( y.addAug(2), std::range_error);
}

TEST(SO2, Inverse){
    InEKF::SO2<> x(PI/2);

    EXPECT_MATRICES_EQ(x.inverse()(), x().inverse());
}

TEST(SO2, Exp){
    Eigen::VectorXd x(3);
    x << 1,2,3;

    InEKF::SO2<Eigen::Dynamic> ours = InEKF::SO2<Eigen::Dynamic>::Exp(x);
    Eigen::Matrix2d theirs = InEKF::SO2<Eigen::Dynamic>::Wedge(x).exp();

    EXPECT_MATRICES_EQ(ours(), theirs);
    EXPECT_MATRICES_EQ(ours.Aug(), x.tail(2));
}

TEST(SO2, Log){
    Eigen::VectorXd x(3);
    x << 0, 1, 2;
    InEKF::SO2<Eigen::Dynamic> state(x);

    Eigen::Vector3d xi = state.log();
    EXPECT_MATRICES_EQ(xi, x);
}

TEST(SO2, Wedge){
    Eigen::VectorXd x(3);
    x << 1, 2, 3;

    Eigen::Matrix2d ours = InEKF::SO2<Eigen::Dynamic>::Wedge(x); 
    Eigen::Matrix2d theirs;
    theirs << 0, -1,
                1, 0;

    EXPECT_MATRICES_EQ(ours, theirs);
}

TEST(SO2, Adjoint){
    InEKF::SO2<1> x;
    EXPECT_MATRICES_EQ(x.Ad(), Eigen::Matrix2d::Identity());
    EXPECT_MATRICES_EQ(InEKF::SO2<1>::Ad(x), Eigen::Matrix2d::Identity());
}