#!/usr/bin/env python3

# system imports
from os import stat
import numpy as np
import math

# ROS imports
import rospy
from geometry_msgs.msg import Point
from geographic_msgs.msg import GeoPoseStamped
from vrx_gazebo.msg import Task

# local imports
from robowalrus.msg import VRXState
from tools.heading_calculator import HeadingCalculator
from State import State
#import tools.enable_table as ENABLE
from tools import enable_table

class TrajectoryGenerationTask1Node:
    '''
    :class TrajectoryGenerationTask1Node: Outputs offset point for robot to go to
        for task 1.

    '''

    def __init__(self):
        '''
        initialization function
        '''
        self.has_offset = False
        self.has_goal_state = False
        self.msg_formulated = False
        self.offset = np.array([0.0, 0.0])
        self.goal_msg = VRXState()
        self.adj_goal_msg = VRXState()
        self.goal_data = None
        self.heading_calculator = HeadingCalculator(5)
        self.estimated_state = State(0, 0, 0)
        self.enabled = False
        self.setup = False
        self.task_info_sub = rospy.Subscriber("/vrx/task/info", Task, self.task_info_callback, queue_size=1)

    def task_info_callback(self, msg):
        if self.setup:
            return
        elif msg.name in enable_table.TRAJECTORY_GENERATION_TASK1:
            self.enabled = True
            self.setup = True
            self.setup_pubs_subs()

    def setup_pubs_subs(self):
        # Initialize ROS
        self.desired_state_pub = rospy.Publisher("/vrx_controller/trajectory", VRXState, queue_size=1)
        rospy.Subscriber("/vrx_controller/lateral_offset", Point, self.offset_callback)
        rospy.Subscriber("/vrx/station_keeping/goal", GeoPoseStamped, self.goal_callback)
        rospy.Subscriber("/vrx_controller/estimated_state", VRXState, self.state_callback)
        rospy.Timer(rospy.Duration(.1), self.timer_callback)

    def offset_callback(self, data):
        '''
        offset topic callback function. Updates gloabl offset variables and formulates 
            trajectory msg if it has goal information already

        :param data: contains x and y offset information
        :type data: Point ROS msg

        '''
        if self.has_offset:
            return
        self.offset = np.array([data.x, data.y])
        print(f'offset: {self.offset}')
        self.has_offset = True
        if self.has_goal_state:
            self.formulate_msg(self.goal_data)

    def goal_callback(self, data):
        '''
        goal topic callback function. Updates global goal_data variable and formulates 
            trajectory msg if the offset information has already been received

        :param data: contains longitude and latitude point for WAMV to go to
        :type data: GeoPoseStamped ROS msg

        '''
        if self.has_goal_state:
            return
        print(f'goal state: {data}')
        if self.has_offset:
            self.formulate_msg(data)
        self.goal_data = data
        self.has_goal_state = True

    def formulate_msg(self, data):
        '''
        Performs offset calculations so that the generator knows the local x, y coordinates for
            the vehicle to go to.

        :param data: contains longitude and latitude point for WAMV to go to
        :type data: GeoPoseStamped ROS msg

        '''
        if self.msg_formulated:
            return
        self.init_lat = float(self.offset[1])
        self.init_long = float(self.offset[0])
        self.goal_msg.x, self.goal_msg.y = self.convert_to_cartesian(data.pose.position.latitude, data.pose.position.longitude)
        print(f'init_lat: {self.init_lat}')
        print(f'init_long: {self.init_long}')
        print(f'self.goal_msg.x: {self.goal_msg.x}')
        print(f'self.goal_msg.y: {self.goal_msg.y}')
        self.adj_goal_msg.x = self.goal_msg.x
        self.adj_goal_msg.y = self.goal_msg.y

        # convert goal heading to psi from the given quaternion
        orientation = data.pose.orientation
        t0 = +2.0 * (orientation.w * orientation.z +
                    orientation.x * orientation.y)
        t1 = +1.0 - 2.0 * (orientation.y * orientation.y +
                            orientation.z * orientation.z)
        self.goal_msg.psi = math.atan2(t0, t1)
        self.adj_goal_msg.psi = self.goal_msg.psi
        self.heading_calculator.update_waypoint(self.goal_msg.x, self.goal_msg.y, self.goal_msg.psi)
        self.msg_formulated = True
        print(self.goal_msg.psi)
        print(self.goal_msg.x)
        print(self.goal_msg.y)

    def convert_to_cartesian(self, lat, long):
        """
        This converts the given lat, long, and alt parameters to cartesian coordinates (x, y, z)
        In frame ENU.

        :param lat: Latitude given from GPS
        :param long: Longitude given from GPS
        :param alt: Altitude given from GPS

        :return: x, and y coordinates

        """
        lat_rad = lat*np.pi/180
        long_rad = long*np.pi/180
        std_parallel = self.init_lat
        globe_radius = 6370.0*pow(10, 3)
        print(f'lat_rad: {lat_rad}')
        print(f'self.init_lat: {self.init_lat}')
        y = globe_radius * (lat_rad - self.init_lat)
        x = globe_radius * (long_rad - self.init_long) * np.cos(std_parallel)
        return x, y

    def timer_callback(self, time_msg):
        '''
        Publishes goal pose as long as the msg has been properly formulated

        :param time_msg: contains timing info
        :type time_msg: Timer ROS msg

        '''
        if self.msg_formulated:
            self.adj_goal_msg.psi = self.heading_calculator.update_position(self.estimated_state.x, self.estimated_state.y)
            self.desired_state_pub.publish(self.adj_goal_msg)

    def state_callback(self, state_msg):
        self.estimated_state = State(state_msg.x, state_msg.y, state_msg.psi)

    def run(self):
        """
        This loop runs until ROS node is killed. Prevents callbacks from being called concurrently
        """
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('trajectory_generation_task1_node', anonymous=True)
    try:
        ros_node = TrajectoryGenerationTask1Node()
        ros_node.run()
    except rospy.ROSInterruptException:
        pass