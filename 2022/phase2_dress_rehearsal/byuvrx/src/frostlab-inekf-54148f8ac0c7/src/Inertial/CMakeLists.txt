cmake_minimum_required(VERSION 3.11)

# find all source and header files
set(SRCS InertialProcess.cpp
          DVLSensor.cpp
          DepthSensor.cpp)

# Create Library
set(LIB_NAME Inertial)
add_library(${LIB_NAME} "${SRCS}") 
add_library(${PROJECT_NAME}::${LIB_NAME} ALIAS ${LIB_NAME})

# Set Library Properties
target_include_directories(${LIB_NAME} PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  )  
target_compile_features(${LIB_NAME} PUBLIC cxx_std_14)
target_link_libraries(${LIB_NAME} PUBLIC Eigen3::Eigen InEKF::Core)