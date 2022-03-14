#include "gtest/gtest.h"
#include <Eigen/Core>
#include <InEKF/Core>
#include <InEKF/Inertial>

#define EXPECT_MATRICES_EQ(M_actual, M_expected) \
  EXPECT_TRUE(M_actual.isApprox(M_expected, 1e-6)) << "  Actual:\n" << M_actual << "\nExpected:\n" << M_expected


TEST(InertialProcess, f){
    InEKF::SE3<2,6> state;
    Eigen::Matrix<double,6,1> u;
    u << 1,1,1,1,1,1+9.81;

    InEKF::InertialProcess ip;
    InEKF::SE3<2,6> result = ip.f(u, 1, state);

    EXPECT_MATRICES_EQ(result.R()(), InEKF::SO3<>::Exp(u.head(3))());
    EXPECT_MATRICES_EQ(result[0], Eigen::Vector3d::Ones());
    EXPECT_MATRICES_EQ(result[1], Eigen::Vector3d::Ones()/2);
}

// TODO: Find a good way to test makePhi?