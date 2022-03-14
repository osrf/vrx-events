FROM quay.io/pypa/manylinux2014_x86_64

# Install packages required for build
RUN yum install -y devtoolset-7-gcc-c++

# Download and install eigen
RUN git clone https://gitlab.com/libeigen/eigen.git
RUN cd eigen && mkdir build \
            && cd build \
            && cmake .. \
            && make install