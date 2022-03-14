#include "gtest/gtest.h"
#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>
#include <InEKF/Core>

#define EXPECT_MATRICES_EQ(M_actual, M_expected) \
  EXPECT_TRUE(M_actual.isApprox(M_expected, 1e-6)) << "  Actual:\n" << M_actual << "\nExpected:\n" << M_expected

namespace Eigen{
    typedef Matrix<double,5,5> Matrix5d;
}

TEST(SE3, BaseConstructor1){
    Eigen::Matrix5d state = Eigen::Matrix5d::Identity();
    Eigen::Matrix<double,11,11> sigma = Eigen::Matrix<double,11,11>::Identity();
    Eigen::Vector2d aug = Eigen::Vector2d::Ones();

    InEKF::SE3<2,2> x(state, sigma, aug);

    EXPECT_MATRICES_EQ(state, x());
    EXPECT_MATRICES_EQ(sigma, x.Cov());
    EXPECT_MATRICES_EQ(aug, x.Aug());
    EXPECT_TRUE(x.Uncertain());
}

TEST(SE3, BaseConstructor2){
    Eigen::MatrixXd state = Eigen::Matrix5d::Identity();
    Eigen::Matrix<double,10,10> sigma = Eigen::Matrix<double,10,10>::Identity();
    Eigen::Matrix<double,1,1> aug = Eigen::Matrix<double,1,1>::Ones();

    // make sure it gets mad if we pass the wrong sigma
    EXPECT_THROW(InEKF::SE3<Eigen::Dynamic> x(state, sigma);, std::range_error);
    // Or doesn't if we pass the right one
    EXPECT_NO_THROW(  (InEKF::SE3<Eigen::Dynamic,1>(state, sigma, aug)) );
    // dynamic aug also works
    Eigen::Matrix5d state2 = Eigen::Matrix5d::Identity();
    EXPECT_NO_THROW(  (InEKF::SE3<2,Eigen::Dynamic>(state2, sigma, aug)) );
}

TEST(SE3, TangentConstructor1){
    Eigen::Matrix<double,10,1> x = Eigen::Matrix<double,10,1>::LinSpaced(10,0,10);

    InEKF::SE3<2,1> state(x);
    EXPECT_MATRICES_EQ(state.R()(), InEKF::SO3<>::Exp(x.head(3))() );
    EXPECT_MATRICES_EQ(state[0], x.segment(3,3));
    EXPECT_MATRICES_EQ(state[1], x.segment(6,3));
    EXPECT_MATRICES_EQ(state.Aug(), x.tail(1));
}

TEST(SE3, TangentConstructor2){
    Eigen::VectorXd x = Eigen::Matrix<double,10,1>::LinSpaced(10,0,10);


    InEKF::SE3<Eigen::Dynamic,1> state(x);
    EXPECT_MATRICES_EQ(state.R()(), InEKF::SO3<>::Exp(x.head(3))() );
    EXPECT_MATRICES_EQ(state[0], x.segment(3,3));
    EXPECT_MATRICES_EQ(state[1], x.segment(6,3));
    EXPECT_MATRICES_EQ(state.Aug(), x.tail(1));

    InEKF::SE3<2,Eigen::Dynamic> state2(x);
    EXPECT_MATRICES_EQ(state2.R()(), InEKF::SO3<>::Exp(x.head(3))() );
    EXPECT_MATRICES_EQ(state2[0], x.segment(3,3));
    EXPECT_MATRICES_EQ(state2[1], x.segment(6,3));
    EXPECT_MATRICES_EQ(state2.Aug(), x.tail(1));

    EXPECT_THROW( (InEKF::SE3<Eigen::Dynamic,Eigen::Dynamic>(x)), std::range_error);
}

TEST(SE3, PlainConstructor){
    InEKF::SE3<> x(0,0,0,4,5,6);
    EXPECT_MATRICES_EQ(x.R()(), Eigen::Matrix3d::Identity());
    EXPECT_EQ(x()(0,3), 4);
    EXPECT_EQ(x()(1,3), 5);
    EXPECT_EQ(x()(2,3), 6);
}

TEST(SE3, AddCol){
    InEKF::SE3<Eigen::Dynamic> x;
    EXPECT_MATRICES_EQ(x(), Eigen::Matrix4d::Identity());

    x.addCol(Eigen::Vector3d::Ones(3));

    EXPECT_EQ(x()(0,4), 1);
    EXPECT_EQ(x()(1,4), 1);
    EXPECT_EQ(x()(2,4), 1);

    // TODO: Test adding to Cov

    InEKF::SE3<> y;
    EXPECT_THROW( y.addCol(Eigen::Vector3d::Ones()), std::range_error);
}

TEST(SE3, AddAug){
    InEKF::SE3<1,Eigen::Dynamic> x;
    EXPECT_EQ(x.Aug().rows(), 0);

    x.addAug(2);

    EXPECT_EQ(x.Aug()(0), 2);

    // TODO: Test adding to Cov

    InEKF::SE3<> y;
    EXPECT_THROW( y.addAug(2), std::range_error);
}

TEST(SE3, Inverse){
    InEKF::SE3<> x(1,2,3,4,5,6);
    EXPECT_MATRICES_EQ(x.inverse()(), x().inverse());
}

TEST(SE3, Exp){
    Eigen::Matrix<double,10,1> x = Eigen::Matrix<double,10,1>::LinSpaced(10,0,10);
    
    InEKF::SE3<2,1> ours = InEKF::SE3<2,1>::Exp(x);
    Eigen::Matrix5d theirs = InEKF::SE3<2,1>::Wedge(x).exp(); 
    
    EXPECT_MATRICES_EQ(ours(), theirs);
    EXPECT_MATRICES_EQ(ours.Aug(), x.tail(1));
    EXPECT_THROW((InEKF::SE3<Eigen::Dynamic,2>::Exp(x)), std::range_error);
}

TEST(SE3, Log){
    Eigen::Matrix<double,6,1> xi;
    xi << .1, .2, .3, 4, 5, 6;
    InEKF::SE3<> x = InEKF::SE3<>::Exp(xi);

    EXPECT_MATRICES_EQ(x.log(), xi);
}

TEST(SE3, Wedge){
    Eigen::Matrix<double,10,1> x = Eigen::Matrix<double,10,1>::LinSpaced(10,1,10);

    Eigen::Matrix5d ours = InEKF::SE3<2,1>::Wedge(x); 
    Eigen::Matrix5d theirs;
    theirs << 0, -3,  2,  4,  7,
              3,  0, -1,  5,  8,
             -2,  1,  0,  6,  9,
              0,  0,  0,  0,  0,
              0,  0,  0,  0,  0;

    EXPECT_MATRICES_EQ(ours, theirs);
    EXPECT_THROW((InEKF::SE3<Eigen::Dynamic,2>::Wedge(x)), std::range_error);
}

TEST(SE3, Adjoint){
    Eigen::Matrix<double,10,1> y = Eigen::Matrix<double,10,1>::LinSpaced(10,1,10);
    InEKF::SE3<2,1> x(y);
    Eigen::Matrix<double,10,10> ad = x.Ad();

    Eigen::Matrix3d top;

    for(int i=0;i<3;i++){
        EXPECT_MATRICES_EQ(ad.block(3*i,3*i,3,3), x.R()());
    }
    for(int i=0;i<2;i++){
        top = InEKF::SO3<>::Wedge(x[i])*x.R()();
        EXPECT_MATRICES_EQ(ad.block(3+3*i,0,3,3), top);
    }
    EXPECT_EQ(ad(9,9), 1);
}