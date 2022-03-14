#include "gtest/gtest.h"
#include <Eigen/Core>
#include <InEKF/Core>

#define EXPECT_MATRICES_EQ(M_actual, M_expected) \
  EXPECT_TRUE(M_actual.isApprox(M_expected, 1e-6)) << "  Actual:\n" << M_actual << "\nExpected:\n" << M_expected

// TODO: Test with dynamic types?
using Group = InEKF::SE2<2,1>;
using MatrixH = InEKF::GenericMeasureModel<Group>::MatrixH;
using MatrixS = InEKF::GenericMeasureModel<Group>::MatrixS;
using VectorV = InEKF::GenericMeasureModel<Group>::VectorV;
using VectorB = InEKF::GenericMeasureModel<Group>::VectorB;

TEST(GenericMeasureModel, bConstructor){
    // make b, H and M
    VectorB b;
    b << 0, 0, 0, 1;
    MatrixH H = MatrixH::Zero();
    H.block(0,3,2,2) = Eigen::Matrix2d::Identity();
    MatrixS M = MatrixS::Identity();

    // Try with left error
    InEKF::GenericMeasureModel<Group> l(b, M, InEKF::LEFT);
    EXPECT_MATRICES_EQ(H, l.getH());

    // Try with right error
    InEKF::GenericMeasureModel<Group> r(b, M, InEKF::RIGHT);
    EXPECT_MATRICES_EQ((-1*H), r.getH());

    b(0) = 1;
    EXPECT_THROW( (InEKF::GenericMeasureModel<Group>(b, M, InEKF::LEFT)), std::range_error);
}

TEST(GenericMeasureModel, processZ){
    // make b, H and M
    VectorB b;
    b << 0, 0, 0, 1;
    MatrixS M = MatrixS::Identity();
    Group state;
    InEKF::GenericMeasureModel<Group> S(b, M, InEKF::LEFT);

    VectorV z;
    z << 2,2;

    VectorB expected;
    expected << 2,2,0,1;

    EXPECT_MATRICES_EQ(S.processZ(z, state), expected);
}