set(TESTS test_InertialProcess.cpp
            test_DepthSensor.cpp
            test_DVLSensor.cpp)

set(TEST_LIBS Eigen3::Eigen 
                InEKF::Core 
                InEKF::Inertial)

add_and_install_tests("${TESTS}" "${TEST_LIBS}")