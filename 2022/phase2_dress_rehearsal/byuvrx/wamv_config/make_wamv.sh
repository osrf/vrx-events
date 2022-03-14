#!/bin/bash

DIR="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
echo "HEY"
echo `ls $DIR/component_config.yaml`
roslaunch vrx_gazebo generate_wamv.launch thruster_yaml:=$DIR/thruster_config.yaml component_yaml:=$DIR/component_config.yaml wamv_target:=$DIR/wamv.urdf