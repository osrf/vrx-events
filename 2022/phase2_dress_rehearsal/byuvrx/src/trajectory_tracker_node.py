#!/usr/bin/env python3

# system imports
import numpy as np

# ROS imports
import rospy
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Wrench

# Local imports
from robowalrus.msg import VRXState, ForceHeading
from tools.pid import PID
from State import State
from vrx_gazebo.msg import Task
#import tools.enable_table as ENABLE
from tools import enable_table

class TrajectoryTrackerNode():
    '''
    :class TrajectoryTrackerNode: PID Controller that calculates force needed
        to achieve a desired state.

    '''

    def __init__(self):
        '''
        initialization function
        '''
        # Initialize private variables
        kp_x = rospy.get_param('trajectory_tracker/gains/kp_x')
        ki_x = rospy.get_param('trajectory_tracker/gains/ki_x')
        kd_x = rospy.get_param('trajectory_tracker/gains/kd_x')
        kp_y = rospy.get_param('trajectory_tracker/gains/kp_y')
        ki_y = rospy.get_param('trajectory_tracker/gains/ki_y')
        kd_y = rospy.get_param('trajectory_tracker/gains/kd_y')
        anti_windup_x = rospy.get_param('trajectory_tracker/gains/anti_windup_x')
        anti_windup_y = rospy.get_param('trajectory_tracker/gains/anti_windup_y')

        Ts = rospy.get_param('Ts')

        self.pid_x = PID(kp_x, ki_x, kd_x, Ts, anti_windup_derivative_max=anti_windup_x)
        self.pid_y = PID(kp_y, ki_y, kd_y, Ts, anti_windup_derivative_max=anti_windup_y)

        self.state = State(rospy.get_param('trajectory_tracker/init_state/x_init'), 
            rospy.get_param('trajectory_tracker/init_state/y_init'),
            rospy.get_param('trajectory_tracker/init_state/psi_init'))
        self.state_dot = State(rospy.get_param('trajectory_tracker/init_state/xdot_init'), 
            rospy.get_param('trajectory_tracker/init_state/ydot_init'),
            rospy.get_param('trajectory_tracker/init_state/psidot_init'))

        self.state_c = State(rospy.get_param('trajectory_tracker/init_commanded/xc_init'),
            rospy.get_param('trajectory_tracker/init_commanded/yc_init'),
            rospy.get_param('trajectory_tracker/init_commanded/psic_init'))
        self.state_dot_c = State(rospy.get_param('trajectory_tracker/init_commanded/xdotc_init'),
            rospy.get_param('trajectory_tracker/init_commanded/ydotc_init'),
            rospy.get_param('trajectory_tracker/init_commanded/psidotc_init'))

        self.force_c = Vector3(0.0, 0.0, 0.0)
        self.f_max_forward = 500
        self.f_max_backward = 200

        self.enabled = False
        self.setup = False
        rospy.Subscriber("/vrx/task/info", Task, self.task_info_callback)
        
    def task_info_callback(self, msg):
        if self.setup:
            return
        elif msg.name in enable_table.TRAJECTORY_TRACKER_NODE:
            self.enabled = True
            self.setup = True
            self.setup_pubs_subs()

    def setup_pubs_subs(self):
        self.force_heading_pub = rospy.Publisher("/vrx_controller/force_heading_d", ForceHeading, queue_size=1)

        rospy.Subscriber("/vrx_controller/estimated_state", VRXState, self.state_callback)
        rospy.Subscriber("/vrx_controller/trajectory", VRXState, self.trajectory_callback)

        # only used in testing
        self.wrench_d_pub = rospy.Publisher("/vrx_controller/wrench_d", Wrench, queue_size=1)

        self.testing = False

    def run(self):
        '''
        This loop runs until ROS node is killed
        '''
        rospy.spin()

    def state_callback(self, msg):
        '''
        state_estimation topic callback function. Updates global state variables

        :param msg: contains state information
        :type msg: VRXState Ros msg

        '''
        self.state = State(msg.x, msg.y, msg.psi)
        self.state_dot = State(msg.xdot, msg.ydot, msg.psidot)

    def trajectory_callback(self, msg):
        '''
        trajectory topic callback function. Triggers the program to calculate
            force needed to achieve desired state.

        :param msg: contains commanded state information
        :type msg: VRXState Ros msg

        '''
        self.state_c = State(msg.x, msg.y, msg.psi)
        self.state_dot_c = State(msg.xdot, msg.ydot, msg.psidot)

        R = np.array([[np.cos(self.state.psi), np.sin(self.state.psi)],
                      [-np.sin(self.state.psi), np.cos(self.state.psi)]])

        state_b = R @ np.array([[self.state.x], [self.state.y]])
        state_dot_b = R @ np.array([[self.state_dot.x], [self.state_dot.y]])
        state_c_b = R @ np.array([[self.state_c.x], [self.state_c.y]])

        F_x = self.pid_x.update(np.array([[state_b.item(0)], [state_dot_b.item(0)]]), state_c_b.item(0)) 
        F_y = self.pid_y.update(np.array([[state_b.item(1)], [state_dot_b.item(1)]]), state_c_b.item(1)) 

        # Limit the force so that it doesn't outweigh the torque the vehicle is trying to produce
        theta = np.arctan2(F_y, F_x)
        f_max = ((self.f_max_backward - self.f_max_forward) / np.pi) * theta + self.f_max_forward
        f_mag = np.sqrt(F_x**2 + F_y**2)
        if abs(f_mag) > abs(f_max):
            scaling_factor = abs(f_max / f_mag)
            F_x *= scaling_factor
            F_y *= scaling_factor
        # print(f'f_max: {f_max}')
        # print(f'F_x: {F_x}')
        # print(f'F_y: {F_y}')


        # Rotates F_x and F_y out of inertial frame and into body frame
        # F_body = \
        #      @ \
        #     np.array([[F_x], [F_y]])
        self.force_c = Vector3(F_x, F_y, 0.0)
        self.psi_c = msg.psi
        self.publish_force_heading()

        # print statements can be useful when debugging to see where vehicle is /
        # where it should be going
        # print(self.state_c)
        # print(self.state)

    def publish_force_heading(self):
        '''
        Publishes commanded force and heading
        '''
        # Publishes to heading control if running with heading control,
        # publishes straight to control allocation if not
        if not self.testing:
            msg = ForceHeading()
            msg.force = self.force_c
            msg.heading = self.psi_c
            self.force_heading_pub.publish(msg)
        else:
            wrench_msg = Wrench()
            wrench_msg.force = self.force_c
            self.wrench_d_pub.publish(wrench_msg)


if __name__ == '__main__':
    rospy.init_node('trajectory_tracker_node', anonymous=True)
    try:
        ros_node = TrajectoryTrackerNode()
        ros_node.run()
    except rospy.ROSInterruptException:
        pass
