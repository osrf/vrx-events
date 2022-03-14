#!/usr/bin/env python3

# system imports
import numpy as np
import math

# ROS imports
import rospy
from geometry_msgs.msg import Point
from geographic_msgs.msg import GeoPath
from tools.heading_calculator import HeadingCalculator
from State import State
from vrx_gazebo.msg import Task

# local imports
from robowalrus.msg import VRXState
#import tools.enable_table as ENABLE
from tools import enable_table

WEIGHTING_TERM = 0.75

class TrajectoryGenerationTask2Node:
    '''
    :class TrajectoryGenerationTask2Node: Outputs offset points for robot to go to
        for task 2.

    '''

    def __init__(self):
        '''
        initialization function
        '''

        # Offset and goal variables
        self.has_offset = False
        self.has_goal_state = False
        self.msg_formulated = False
        self.offset = np.array([0.0, 0.0])
        self.goal_msg = VRXState()
        self.goal_data = None
        self.task_info = Task()

        # Estimated state
        self.estimated_state = State(0, 0, 0)

        # Waypoint variables
        self.waypoints = list()
        self.has_waypoints = False
        self.current_waypoint_index = 0 # change to index of the closest waypoint
        self.minimum_current_pose_error = 100
        self.desired_pose_error = 0.5
        self.time_per_waypoint = -1
        self.time_remaining_per_waypoint = -1
        self.heading_calculator = HeadingCalculator(radius=5)
        
        self.inc_time_taken = False
        self.enabled = False
        self.setup = False
        self.task_info_sub = rospy.Subscriber("/vrx/task/info", Task, self.task_info_callback, queue_size=1)

    def task_info_callback(self, msg):
        self.task_info = msg
        if self.setup:
            return
        elif msg.name in enable_table.TRAJECTORY_GENERATION_TASK2:
            self.enabled = True
            self.setup = True
            self.setup_pubs_subs()

    def setup_pubs_subs(self):
        # Initialize ROS
        # Ros publishers and subscribers
        self.desired_state_pub = rospy.Publisher("/vrx_controller/trajectory", VRXState, queue_size=1)
        rospy.Subscriber("/vrx_controller/lateral_offset", Point, self.offset_callback)
        rospy.Subscriber("/vrx/wayfinding/waypoints", GeoPath, self.goal_callback)
        rospy.Subscriber("/vrx_controller/estimated_state", VRXState, self.estimated_state_callback)
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
            self.make_waypoints(self.goal_data)

    def goal_callback(self, data):
        '''
        goal topic callback function. Updates global goal_data variable and formulates 
            trajectory msg if the offset information has already been received

        :param data: contains longitude and latitude point for WAMV to go to
        :type data: GeoPoseStamped ROS msg

        '''
        if self.has_goal_state:
            return
        print(f'goal poses: {data.poses}')
        if self.has_offset:
            self.make_waypoints(data.poses)
        self.goal_data = data.poses
        self.has_goal_state = True

    def estimated_state_callback(self, data):
        '''
        gets the estimated state so that we can calculate the pose error

        :param data: contains estimated x, y, psi and their derivatives
        :type data: VRXState msg
        '''
        self.estimated_state = State(data.x, data.y, data.psi)

    def sort_waypoints(self):
        '''
        Sorts the waypoints so that the vehicle always goes to the closest unvisited waypoint next.

        Updates the global variable self.waypoints.
        '''
        sorted_waypoint_list = list()
        closest_distance = 9999 # assume that waypoints are within 9999 m of vehicle position and of each other
        closest_waypoint_index = -1

        # move the waypoint closest to the vehicle, from self.waypoints to the sorted list
        for i in range(len(self.waypoints)):
            euclidean_distance = np.sqrt(pow(self.waypoints[i][0] - self.estimated_state.x, 2) + pow(self.waypoints[i][1] - self.estimated_state.y, 2))
            if euclidean_distance < closest_distance:
                closest_distance = euclidean_distance
                closest_waypoint_index = i
        sorted_waypoint_list.append(self.waypoints[closest_waypoint_index])
        self.waypoints.pop(closest_waypoint_index)
        
        # move all the other waypoints in order
        for i in range(len(self.waypoints)): # used to get the last item in the sorted waypoint list
            closest_distance = 9999
            closest_waypoint_index = -1
            for j in range(len(self.waypoints)):
                euclidean_distance = np.sqrt(pow(self.waypoints[j][0] - sorted_waypoint_list[i][0], 2) + pow(self.waypoints[j][1] - sorted_waypoint_list[i][1], 2))
                if euclidean_distance < closest_distance:
                    closest_distance = euclidean_distance
                    closest_waypoint_index = j
            sorted_waypoint_list.append(self.waypoints[closest_waypoint_index])
            self.waypoints.pop(closest_waypoint_index)

        self.waypoints = sorted_waypoint_list

    def make_waypoints(self, data):
        '''
        Performs offset calculations so that the generator knows the local x, y coordinates for
            the vehicle to go to.

        :param data: contains longitude and latitude point for WAMV to go to
        :type data: GeoPoseStamped ROS msg

        '''
        if self.has_waypoints:
            return
        self.init_lat = float(self.offset[1])
        self.init_long = float(self.offset[0])
        print('data here')
        for waypoint in data:
            x, y = self.convert_to_cartesian(waypoint.pose.position.latitude, waypoint.pose.position.longitude)
            # convert goal heading to psi from the given quaternion
            orientation = waypoint.pose.orientation
            t0 = +2.0 * (orientation.w * orientation.z +
                orientation.x * orientation.y)
            t1 = +1.0 - 2.0 * (orientation.y * orientation.y +
                        orientation.z * orientation.z)
            psi = math.atan2(t0, t1)
            self.waypoints.append([x, y, psi])
        self.sort_waypoints()
        self.current_waypoint_index = 0
        self.goal_msg.x = self.waypoints[0][0]
        self.goal_msg.y = self.waypoints[0][1]
        self.goal_msg.psi = self.waypoints[0][2]
        self.heading_calculator.update_waypoint(self.goal_msg.x, self.goal_msg.y, self.goal_msg.psi)
        self.has_waypoints = True
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

    def get_current_pose_error(self):
        """
        This gets the current instantaneous pose error of the vehicle.

        :return: vehicle pose error
        """
        if self.has_waypoints:
            euclidean_distance = np.sqrt(pow(self.goal_msg.x - self.estimated_state.x, 2) + pow(self.goal_msg.y - self.estimated_state.y, 2))
            pose_error = euclidean_distance + pow(WEIGHTING_TERM, euclidean_distance) * abs(self.goal_msg.psi - self.estimated_state.psi)
            return pose_error
        return 100


    def timer_callback(self, time_msg):
        '''
        Publishes goal pose as long as the msg has been properly formulated

        :param time_msg: contains timing info
        :type time_msg: Timer ROS msg

        '''
        if self.inc_time_taken:
            self.time_taken_on_waypoint += .1
        if self.has_waypoints:
            # print('sending')
            self.check_if_ready_for_next_waypoint()
            adj_goal_msg = VRXState()
            adj_goal_msg.x = self.goal_msg.x
            adj_goal_msg.y = self.goal_msg.y
            adj_goal_msg.psi = self.heading_calculator.update_position(self.estimated_state.x, self.estimated_state.y)
            self.desired_state_pub.publish(adj_goal_msg)

    def get_remaining_distance(self):
        '''
        Gets the distance needed to travel from current waypoint to get to every remaining waypoint

        :return: distance of path visiting all remaining waypoints
        '''
        remaining_distance = 0
        for i in range(len(self.waypoints) - 1):
            if i < self.current_waypoint_index:
                continue
            euclidean_distance = np.sqrt(pow(self.waypoints[i][0] - self.waypoints[i + 1][0], 2) + pow(self.waypoints[i][1] - self.waypoints[i + 1][1], 2))
            remaining_distance += euclidean_distance
        return remaining_distance

    def check_if_ready_for_next_waypoint(self):
        if self.task_info.state != "running" or self.task_info.remaining_time.secs == 0:
            return
        #calculate desired pose error based on remaining time, remaining waypoints, and remaining distance
        if self.time_per_waypoint == -1:
            distance_to_next_waypoint = np.sqrt(pow(self.waypoints[self.current_waypoint_index][0] - self.estimated_state.x, 2) + pow(self.waypoints[self.current_waypoint_index][1] - self.estimated_state.y, 2))
            #self.time_per_waypoint = self.task_info.remaining_time.secs * distance_to_next_waypoint / (self.get_remaining_distance() + distance_to_next_waypoint)
            self.time_per_waypoint = self.task_info.remaining_time.secs / (len(self.waypoints) - self.current_waypoint_index)
            self.time_taken_on_waypoint = 0
            self.inc_time_taken = True
            print(f'distance: {distance_to_next_waypoint}')
            print(f'time remaining: {self.task_info.remaining_time.secs}')
            print(f'time allocated for waypoint: {self.time_per_waypoint}')
        if (self.time_per_waypoint - self.time_taken_on_waypoint == 0):
            self.desired_pose_error = 50
        else:
            if self.time_taken_on_waypoint < self.time_per_waypoint / 2:
                self.desired_pose_error = .04
            else:
                self.desired_pose_error = 1/pow((self.time_per_waypoint - self.time_taken_on_waypoint),2) + .05
                #self.desired_pose_error = (self.time_remaining_per_waypoint/self.time_taken_on_waypoint)^6 + .05
        #keep track of the minimum current pose error for this waypoint
        if self.get_current_pose_error() < self.minimum_current_pose_error:
            self.minimum_current_pose_error = self.get_current_pose_error()
        if self.minimum_current_pose_error < self.desired_pose_error and self.current_waypoint_index != (len(self.waypoints) - 1):
            print(f'min pose error = {self.minimum_current_pose_error}')
            self.minimum_current_pose_error = 100
            self.time_per_waypoint = -1
            self.time_remaining_per_waypoint = -1
            self.current_waypoint_index += 1
            self.goal_msg.x = self.waypoints[self.current_waypoint_index][0]
            self.goal_msg.y = self.waypoints[self.current_waypoint_index][1]
            self.goal_msg.psi = self.waypoints[self.current_waypoint_index][2]
            self.heading_calculator.update_waypoint(self.goal_msg.x, self.goal_msg.y, self.goal_msg.psi)

    def run(self):
        """
        This loop runs until ROS node is killed. Prevents callbacks from being called concurrently
        """
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('trajectory_generation_task2_node', anonymous=True)
    try:
        ros_node = TrajectoryGenerationTask2Node()
        ros_node.run()
    except rospy.ROSInterruptException:
        pass