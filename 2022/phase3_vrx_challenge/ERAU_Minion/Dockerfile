# This is an auto generated Dockerfile for ros:ros-base
# generated from docker_images/create_ros_image.Dockerfile.em

# install ROS
FROM ros:noetic-ros-base

# Ubuntu arguments
ARG RELEASE=focal
ARG DIST=noetic

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV ROS_DISTRO=${DIST}
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES=graphics
ENV NVIDIA_VISIBLE_DEVICES=all

# timezone
RUN echo 'Etc/UTC' > /etc/timezone \
 && ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime \
 && apt update \
 && apt install -q -y --no-install-recommends \
    tzdata \
 && dpkg-reconfigure --frontend noninteractive tzdata \
 && apt clean \
 && rm -rf /var/lib/apt/lists/*

# install ros packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-ros-base \ 
    ros-noetic-geodesy \ 
    ros-noetic-robot-localization \
    ros-noetic-robot-state-publisher \
    ros-noetic-joint-state-publisher \
    ros-noetic-pid \
    ros-noetic-xacro \
    ros-noetic-gazebo-ros \
    ros-noetic-pcl-* \
    libgeographic-dev \
    gazebo11 \
    rviz \
    build-essential \
    cmake \
    cppcheck \
    curl \
    git \
    gnupg \
    libeigen3-dev \
    libgles2-mesa-dev \
    lsb-release \
    pkg-config \
    protobuf-compiler \
    python3-dbg \
    python3-pip \
    python3-scipy \
    python3-venv \
    qtbase5-dev \
    software-properties-common \
    sudo \
    wget \
&& rm -rf /var/lib/apt/lists/*

# create workspace
RUN mkdir -p /minion_ws/src


# clone vrx from GitHub
RUN cd /minion_ws/src && git clone https://github.com/osrf/vrx.git

# build vrx
RUN . /opt/ros/noetic/setup.sh \
 && cd /minion_ws \
 && catkin_make

# clone packages from GitHub
RUN cd /minion_ws/src && git clone https://github.com/charlestmontagnoli/vrx2022submission.git

# build minion_ws
RUN . /opt/ros/noetic/setup.sh \
 && cd /minion_ws \
 && catkin_make -DCATKIN_WHITELIST_PACKAGES="minion_bugs" \
 && catkin_make -DCATKIN_WHITELIST_PACKAGES="point_cloud_module" \
 && catkin_make -DCATKIN_WHITELIST_PACKAGES="docking" \
 && catkin_make -DCATKIN_WHITELIST_PACKAGES=""

# copy files
COPY ros_entrypoint.sh run_my_system.bash /

ENTRYPOINT ["/ros_entrypoint.sh"]
