#include "gtest/gtest.h"
#include <Eigen/Core>
#include <InEKF/Core>
#include <InEKF/Inertial>

#define EXPECT_MATRICES_EQ(M_actual, M_expected) \
  EXPECT_TRUE(M_actual.isApprox(M_expected, 1e-6)) << "  Actual:\n" << M_actual << "\nExpected:\n" << M_expected


TEST(DepthSensor, processZ){
    InEKF::SE3<2,6> state;
    Eigen::Matrix<double,1,1> z;
    z << 5;

    InEKF::DepthSensor ds;
    Eigen::Matrix<double,5,1> z_full = ds.processZ(z, state);

    Eigen::Matrix<double,5,1> expected;
    expected << 0, 0, 5, 0, 1;

    EXPECT_MATRICES_EQ(z_full, expected);
}

// TODO: Find a way to test calcSInverse?
