# Invariant Extended Kalman Filter
InEKF is a C++ library that implements the Invariant Extend Kalman Filter (InEKF) in a modular to enable easy application to any system.

## Features
- Support for Right & Left filters.
- Base classes provided easy extension via inheritance.
- Coded using static Eigen types for efficient structure.
- Fully featured python interface for use in classroom, testing, etc.
- C++14 and above supported.
- Fullly templated Lie Groups SO2, SO3, SE2, SE3 to enable additional tracking of Euclidean states and multiple extra columns in SE2/SE3.
- Dynamic Lie Groups types to add columns to SE2/SE3 on the fly (for InEKF SLAM).
- Various examples to get started.

## Building & Linking
To use InEKF, only `Eigen` is necessary. This can be installed via `apt-get` or from source. Also, a version of python with numpy and matplotlib installed for plotting reasons (I recommend systemwide, conda gave me fits).

InEKF is built via cmake, and thus can be built in the usual cmake fashion:
```bash
mkdir build
cd build
cmake ..
make
sudo make install
```

This will install a custom target that can be linked via CMake as

```cmake
find_package(Eigen3 CONFIG REQUIRED)
find_package(InEKF CONFIG REQUIRED)
target_link_libraries(mytarget PUBLIC InEKF::Core InEKF::Inertial InEKF::SE2Models)
```

A python wrapper is also [available](python/README.md).

## Structure
InEKF is split into a couple different libraries

### Core
This library includes all the Lie Groups, `InEKF`, and `GenericMeasureModel` classes along with the base classes `MeasureModel`, `ProcessModel`.


### Inertial
This is the implementation of the Lie Group `SE_2(3)` along with an augmented bias state. Along with it are various process/measurement models defined on this group including as of now `DVLSensor`, `DepthSensor`, and `InertialProcess`.

### SE2Models
Exactly what it sounds like, is used for SLAM in SE2. 

## Extending

InEKF is set up so your process/measure models will be an easy extension and continue to function with `InEKF` and `LieGroups` if defined properly. Note this can be done in python or C++. The following is what must be defined/done to successfully do this. The following methods/variables for each base class must be implemented/set

### MeasureModel
All methods are already implemented in the `MeasureModel` class, so overriding them can be decided on a case by case basis. There is a few constants that must be set though.

|        Method         | Use                                                                                                                                      |
| :-------------------: | :--------------------------------------------------------------------------------------------------------------------------------------- |
|       `error_`        | Type of invariant measurement, either `ERROR::LEFT` or `ERROR::Right`                                                                    |
|         `M_`          | Noise parameter. A default should be set in the constructor, and possible a method made to set it                                        |
|         `H_`          | Linearized innovation matrix `H`. Will be hit with adjoint depending on type of filter. Use `H_error_` in `calcSInverse(z, state)`       |
| `processZ(z, state)`  | Any preprocessing that needs to be done on z should be done here. This could include adding 0s and 1s on the end, change of frames, etc. |
|   `calcV(z, state)`   | Accepts an exact size of z, and calculates/returns the innovation. Likely will not need to be overriden.                                 |
| `calcSInverse(state)` | Calculates and returns S^{-1}, the inverse of the measurement covariance. Also likely won't need to be overriden.                        |

### ProcessModel
In contrast, the process model implements a few things that MUST be overriden. 

|         Method          | Use                                                                                               |
| :---------------------: | :------------------------------------------------------------------------------------------------ |
|    `f(u, dt, state)`    | State process model. Returns the state.                                                           |
| `MakePhi(u, dt, state)` | Creates exp(A*dt) to use. Make sure to check what type of error State is and make A accordingly   |
|          `Q_`           | Noise parameter. A default should be set in the constructor, and possible a method made to set it |
