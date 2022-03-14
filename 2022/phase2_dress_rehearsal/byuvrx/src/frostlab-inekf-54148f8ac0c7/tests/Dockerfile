FROM gcc:11

# Install packages required for build
RUN apt-get update
RUN apt-get install -qy cmake python3-dev python3-pip gnuplot

# install eigen
RUN git clone https://gitlab.com/libeigen/eigen.git
RUN cd eigen && mkdir build \
            && cd build \
            && cmake .. \
            && make install

# Install python libraries for testing
RUN pip3 install numpy matplotlib pytest scipy