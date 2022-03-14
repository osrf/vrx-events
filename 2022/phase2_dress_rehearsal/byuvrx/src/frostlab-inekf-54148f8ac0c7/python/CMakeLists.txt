cmake_minimum_required(VERSION 3.16)

include(CMake_Macros.txt)

# setup globals for templating purposes
set(COL "2" CACHE STRING "Number of columns to template")
set(AUG "2" CACHE STRING "Number of augmented states to template")
set(VEC "4" CACHE STRING "Number of vectors controls to template")
# set(PYTHON_VERSION "" CACHE STRING "Python Version to build")
configure_file(${PROJECT_SOURCE_DIR}/python/include/globals.h.in
                ${PROJECT_SOURCE_DIR}/python/include/globals.h)

# Find python
# find_package(Python ${PYTHON_VERSION} EXACT COMPONENTS Interpreter Development REQUIRED)

# Get pybind, if not local, pull from git
find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
  message(STATUS "Pulling pybind11 from git...")
  FetchContent_Declare(
      pybind11
      GIT_REPOSITORY https://github.com/pybind/pybind11.git
      GIT_TAG        v2.6.2
  )
  FetchContent_MakeAvailable(pybind11)
endif()

set(SRCS src/main.cpp 
        src/makeInEKFSE2.cpp
        src/makeLieSO.cpp 
        src/makeLieSE.cpp
        src/makeProcessSO2.cpp
        src/makeProcessSO3.cpp
        src/makeProcessSE2.cpp
        src/makeProcessSE3.cpp
        src/makeMeasureSO.cpp
        src/makeMeasureSE.cpp
        src/makeInertial.cpp
        src/makeSE2Models.cpp
        src/makeInEKFSO2.cpp
        src/makeInEKFSO3.cpp
        src/makeInEKFSE3.cpp
    )

# Setup various python things
set(PYTHON_BUILD_DIRECTORY ${PROJECT_BINARY_DIR}/python)
file(READ "${PROJECT_SOURCE_DIR}/README.md" README_CONTENTS)
configure_file(${PROJECT_SOURCE_DIR}/python/setup.py.in
                ${PYTHON_BUILD_DIRECTORY}/setup.py)

# make linked libraries
set_target_properties(Inertial  PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
set_target_properties(SE2Models PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

# make target and all properties
pybind11_add_module(_inekf "${SRCS}" OPT_SIZE THIN_LTO)

target_link_libraries(_inekf PUBLIC Core Inertial SE2Models)
target_include_directories(_inekf PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/python/include>
  $<INSTALL_INTERFACE:include>
  )  
set_target_properties(_inekf PROPERTIES
  INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib"
  INSTALL_RPATH_USE_LINK_PATH TRUE
  OUTPUT_NAME "_inekf"
  LIBRARY_OUTPUT_DIRECTORY "${PYTHON_BUILD_DIRECTORY}/inekf"
  DEBUG_POSTFIX "" # Otherwise you will have a wrong name
  RELWITHDEBINFO_POSTFIX "" # Otherwise you will have a wrong name
)
target_compile_features(_inekf PUBLIC cxx_std_17)

# setup things to install
add_custom_target(python-install
    COMMAND ${PYTHON_EXECUTABLE} ${PYTHON_BUILD_DIRECTORY}/setup.py install
    WORKING_DIRECTORY ${PYTHON_BUILD_DIRECTORY})
create_symlinks("${CMAKE_CURRENT_SOURCE_DIR}/inekf" "${PYTHON_BUILD_DIRECTORY}/inekf")

# test
if(TESTS)
  add_test(NAME test-python
      COMMAND ${PYTHON_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}
      WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/python
    )
endif()
