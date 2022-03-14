cmake_minimum_required(VERSION 3.11)

# Create Library
set(LIB_NAME Core)
add_library(${LIB_NAME} INTERFACE) 
add_library(${PROJECT_NAME}::${LIB_NAME} ALIAS ${LIB_NAME})

# Set Library Properties
target_include_directories(${LIB_NAME} INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  )  
target_compile_features(${LIB_NAME} INTERFACE cxx_std_14)
target_link_libraries(${LIB_NAME} INTERFACE Eigen3::Eigen)