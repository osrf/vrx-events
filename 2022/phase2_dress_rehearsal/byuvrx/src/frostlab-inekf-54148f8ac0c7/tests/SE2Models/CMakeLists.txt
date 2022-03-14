set(TESTS test_Odometry.cpp)

set(TEST_LIBS Eigen3::Eigen 
                InEKF::Core 
                InEKF::SE2Models)

add_and_install_tests("${TESTS}" "${TEST_LIBS}")