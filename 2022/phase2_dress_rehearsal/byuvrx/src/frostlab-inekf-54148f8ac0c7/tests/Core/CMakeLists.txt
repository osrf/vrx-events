set(TESTS test_SE2.cpp
            test_SE3.cpp
            test_SO2.cpp
            test_SO3.cpp
            test_GenericMeasure.cpp)

set(TEST_LIBS Eigen3::Eigen
                InEKF::Core)

add_and_install_tests("${TESTS}" "${TEST_LIBS}")