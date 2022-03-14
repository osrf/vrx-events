#include "gtest/gtest.h"
#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>
#include <InEKF/Core>

#define EXPECT_MATRICES_EQ(M_actual, M_expected) \
  EXPECT_TRUE(M_actual.isApprox(M_expected, 1e-6)) << "  Actual:\n" << M_actual << "\nExpected:\n" << M_expected


TEST(SE2, BaseConstructor1){
    Eigen::Matrix4d state = Eigen::Matrix4d::Identity();
    Eigen::Matrix<double,7,7> sigma = Eigen::Matrix<double,7,7>::Identity();
    Eigen::Vector2d aug = Eigen::Vector2d::Ones();

    InEKF::SE2<2,2> x(state, sigma, aug);

    EXPECT_MATRICES_EQ(state, x());
    EXPECT_MATRICES_EQ(sigma, x.Cov());
    EXPECT_MATRICES_EQ(aug, x.Aug());
    EXPECT_TRUE(x.Uncertain());
}

TEST(SE2, BaseConstructor2){
    Eigen::MatrixXd state = Eigen::Matrix4d::Identity();
    Eigen::Matrix<double,6,6> sigma = Eigen::Matrix<double,6,6>::Identity();
    Eigen::Matrix<double,1,1> aug = Eigen::Matrix<double,1,1>::Ones();

    // make sure it gets mad if we pass the wrong sigma
    EXPECT_THROW(InEKF::SE2<Eigen::Dynamic> x(state, sigma);, std::range_error);
    // Or doesn't if we pass the right one
    EXPECT_NO_THROW(  (InEKF::SE2<Eigen::Dynamic,1>(state, sigma, aug)) );
    // dynamic aug also works
    Eigen::Matrix4d state2 = Eigen::Matrix4d::Identity();
    EXPECT_NO_THROW(  (InEKF::SE2<2,Eigen::Dynamic>(state2, sigma, aug)) );
}

TEST(SE2, TangentConstructor1){
    Eigen::Matrix<double,6,1> x;
    x << 0, 1, 2, 3, 4, 5;

    InEKF::SE2<2,1> state(x);
    EXPECT_MATRICES_EQ(state.R()(), Eigen::Matrix2d::Identity());
    EXPECT_EQ(state()(0,2), 1);
    EXPECT_EQ(state()(1,2), 2);
    EXPECT_EQ(state()(0,3), 3);
    EXPECT_EQ(state()(1,3), 4);
    EXPECT_EQ(state.Aug()(0), 5);
}

TEST(SE2, TangentConstructor2){
    Eigen::VectorXd x(6);
    x << 0, 1, 2, 3, 4, 5;

    InEKF::SE2<Eigen::Dynamic,1> state(x);
    EXPECT_MATRICES_EQ(state.R()(), Eigen::Matrix2d::Identity());
    EXPECT_EQ(state()(0,2), 1);
    EXPECT_EQ(state()(1,2), 2);
    EXPECT_EQ(state()(0,3), 3);
    EXPECT_EQ(state()(1,3), 4);
    EXPECT_EQ(state.Aug()(0), 5);

    InEKF::SE2<2,Eigen::Dynamic> state2(x);
    EXPECT_MATRICES_EQ(state.R()(), Eigen::Matrix2d::Identity());
    EXPECT_EQ(state2()(0,2), 1);
    EXPECT_EQ(state2()(1,2), 2);
    EXPECT_EQ(state2()(0,3), 3);
    EXPECT_EQ(state2()(1,3), 4);
    EXPECT_EQ(state2.Aug()(0), 5);

    // InEKF::SE2<Eigen::Dynamic,Eigen::Dynamic> state3(x);
    EXPECT_THROW( (InEKF::SE2<Eigen::Dynamic,Eigen::Dynamic>(x)), std::range_error);
}

TEST(SE2, PlainConstructor){
    InEKF::SE2<> x(0,1,2);
    EXPECT_MATRICES_EQ(x.R()(), Eigen::Matrix2d::Identity());
    EXPECT_EQ(x()(0,2), 1);
    EXPECT_EQ(x()(1,2), 2);
}

TEST(SE2, AddCol){
    InEKF::SE2<Eigen::Dynamic> x;
    EXPECT_MATRICES_EQ(x(), Eigen::Matrix3d::Identity());

    x.addCol(Eigen::Vector2d::Ones(2));

    EXPECT_EQ(x()(0,3), 1);
    EXPECT_EQ(x()(1,3), 1);

    // TODO: Test adding to Cov

    InEKF::SE2<> y;
    EXPECT_THROW( y.addCol(Eigen::Vector2d::Ones()), std::range_error);
}

TEST(SE2, AddAug){
    InEKF::SE2<1,Eigen::Dynamic> x;
    EXPECT_EQ(x.Aug().rows(), 0);

    x.addAug(2);

    EXPECT_EQ(x.Aug()(0), 2);

    // TODO: Test adding to Cov

    InEKF::SE2<> y;
    EXPECT_THROW( y.addAug(2), std::range_error);
}

TEST(SE2, Inverse){
    InEKF::SE2<> x(1,1,1);
    EXPECT_MATRICES_EQ(x().inverse(), x.inverse()());
}

TEST(SE2, Exp){
    Eigen::Matrix<double,6,1> x;
    x << 0, 1, 2, 3, 4, 5;
    
    InEKF::SE2<2,1> ours = InEKF::SE2<2,1>::Exp(x);
    Eigen::Matrix4d theirs = InEKF::SE2<2,1>::Wedge(x).exp(); 
    
    EXPECT_MATRICES_EQ(ours(), theirs);
    EXPECT_MATRICES_EQ(ours.Aug(), x.tail(1));
    EXPECT_THROW((InEKF::SE2<Eigen::Dynamic,2>::Exp(x)), std::range_error);
}

TEST(SE2, Log){
    Eigen::Vector3d xi;
    xi << .1, 2, 3;
    InEKF::SE2<> x = InEKF::SE2<>::Exp(xi);

    EXPECT_MATRICES_EQ(x.log(), xi);
}

TEST(SE2, Wedge){
    Eigen::Matrix<double,6,1> x;
    x << 1, 2, 3, 4, 5, 6;

    Eigen::Matrix4d ours = InEKF::SE2<2,1>::Wedge(x); 
    Eigen::Matrix4d theirs;
    theirs << 0, -1, 2, 4,
                1, 0, 3, 5,
                0, 0, 0, 0,
                0, 0, 0, 0;

    EXPECT_MATRICES_EQ(ours, theirs);
    EXPECT_THROW((InEKF::SE2<Eigen::Dynamic,2>::Wedge(x)), std::range_error);
}

TEST(SE2, Adjoint){
    InEKF::SE2<> x(1, 2, 3);
    Eigen::Matrix3d ad = x.Ad();

    EXPECT_MATRICES_EQ(ad.bottomRightCorner(2,2), x.R()());
    EXPECT_DOUBLE_EQ(ad(0,0), 1);
    EXPECT_DOUBLE_EQ(ad(1,0), 3);
    EXPECT_DOUBLE_EQ(ad(2,0), -2);
}