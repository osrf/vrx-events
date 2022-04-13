#!/bin/bash

# Create ros master if not already started
rostopic list > /dev/null 2>&1
retVal=$?
if [ $retVal -ne 0 ]; then
    roscore &
    echo "Wait for 5s to allow rosmaster to start"
    sleep 5s
else
    echo "rosmaster already setup"
fi

source /minion_ws/devel/setup.bash

# Command for running Minion Task Manager
rosrun task_manager task_manager_node
