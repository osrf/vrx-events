#include "gtest/gtest.h"
#include <Eigen/Core>
#include <InEKF/Core>
#include <InEKF/SE2Models>

#define EXPECT_MATRICES_EQ(M_actual, M_expected) \
  EXPECT_TRUE(M_actual.isApprox(M_expected, 1e-6)) << "  Actual:\n" << M_actual << "\nExpected:\n" << M_expected


TEST(OdometryProcess, f){
    InEKF::SE2<> state;
    InEKF::SE2<> U(.1, 1, 2);

    InEKF::OdometryProcess op;

    EXPECT_MATRICES_EQ(op.f(U, 1, state)(), U());
}

TEST(OdometryProcess, MakePhi){
    InEKF::SE2<> state;
    InEKF::SE2<> U(.1, 1, 2);

    InEKF::OdometryProcess op;
    Eigen::Matrix3d Phi;

    // check right
    Phi = op.MakePhi(U, 1, state, InEKF::RIGHT);
    EXPECT_MATRICES_EQ(Phi, Eigen::Matrix3d::Identity());

    // check left
    Phi = op.MakePhi(U, 1, state, InEKF::LEFT);
    EXPECT_MATRICES_EQ(Phi, U.inverse().Ad());
}