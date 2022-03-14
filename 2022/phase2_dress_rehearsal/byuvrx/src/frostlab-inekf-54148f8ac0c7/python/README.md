# Python Wrapper
The python wrapper is made using pybind11. It's a bit heavy to compile due to needing to instantiate so many templates, so the preferred method of installation is via

```
pip install inekf
```

## Building
If you want to build it from source, it's done via the following cmake commands
```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DPYTHON=ON ..
make
make python-install
```

If pybind11 isn't installed locally, CMake will automatically download and compile it. It's recommended to build with no more than 2 cores at a time due to the heavy necessity of instantiated so many templates. If you'd like to set how many LieGroup/MeasureModel/ProcessModel/InEKF templates are done, you can customize it in [the globals header](include/globals.h). And if there's just a few specific ones that are missing, you can add it to the [main compiled file](src/main.cpp).

## Using
The python wrapper can be extended identically in python as you would in C++, see [the README](../README.md) on how to do that. Further, we've provided some syntactic sugar to make swapping between the C++/Python interfaces seamless. For example, in C++ to instantiate an SE3 object with 2 columns, 3 augmented euclidean states would be
```cpp
InEKF::SE3<2,3> x;
```
and in python
```python
x = inekf.SE3[2,3]()
```
The templates of InEKF, the other Lie Groups, and ProcessModels all behave identically.