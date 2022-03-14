cmake_minimum_required(VERSION 3.11)

#################### IMPORT DEPENDENCIES ####################
FetchContent_Declare(
  matplotplusplus
  GIT_REPOSITORY https://github.com/alandefreitas/matplotplusplus
  GIT_TAG v1.1.0
)
FetchContent_MakeAvailable(matplotplusplus)

find_package(Gnuplot REQUIRED)

#################### DEFINE TARGETS + DEP ####################
set(SRCS 
        UnderwaterInertial.cpp
        VictoriaPark.cpp
)

set(LIBS Eigen3::Eigen Core SE2Models Inertial matplot)

#################### MAKE TARGETS ####################
foreach(source_file ${SRCS})
    # make name name of file
    get_filename_component(EXE_NAME ${source_file} NAME_WE)
    add_executable(${EXE_NAME} ${source_file})
    
    # link dependencies
    target_link_libraries(${EXE_NAME} PUBLIC ${LIBS})
    target_include_directories(${EXE_NAME} PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/examples>
    $<INSTALL_INTERFACE:include>
    ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS}
    )  

    # use 17 to remove need for <>
    target_compile_features(${EXE_NAME} PUBLIC cxx_std_17)
endforeach(source_file)