#!/usr/bin/env python3

# system imports
import numpy as np
from numpy.linalg import inv

# ROS imports
import rospy
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Wrench
from vrx_gazebo.msg import Task

# Local imports
from robowalrus.msg import VRXState, ForceHeading
from tools.pid import PID
from State import State
#import tools.enable_table as ENABLE
from tools import enable_table

class HeadingControlNode():
    '''
    :class HeadingControlNode: PID Controller that calculates force needed
        to achieve a desired state.

    '''

    def __init__(self):
        '''
        initialization function
        '''
        # Initialize private variables
        kp = rospy.get_param('heading_control/kp')
        ki = rospy.get_param('heading_control/ki')
        kd = rospy.get_param('heading_control/kd')

        Ts = rospy.get_param('Ts')
        sigma = Ts
        beta = (2.0*sigma-Ts)/(2.0*sigma+Ts)

        self.pid = PID(kp, ki, kd, Ts, dirty_derivative=True, beta=beta)

        self.psi = 0.0
        self.psi_c = 0.0
        self.tau_max = 500.0

        self.force_c = Vector3(0.0, 0.0, 0.0)
        self.testing = True

        self.enabled = False
        self.setup = False
        self.task_info_sub = rospy.Subscriber("/vrx/task/info", Task, self.task_info_callback, queue_size=1)

    def task_info_callback(self, msg):
        if self.setup:
            return
        elif msg.name in enable_table.HEADING_CONTROL_NODE:
            self.enabled = True
            self.setup = True
            self.setup_pubs_subs()

    def setup_pubs_subs(self):
        # Initialize ROS
        rospy.Subscriber("/vrx_controller/force_heading_d", ForceHeading, self.force_heading_callback)
        rospy.Subscriber("/vrx_controller/estimated_state", VRXState, self.state_callback)

        # remove when heading control exists
        self.wrench_d_pub = rospy.Publisher("/vrx_controller/wrench_d", Wrench, queue_size=1)
        
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
        self.psi = msg.psi

    def force_heading_callback(self, msg):
        '''
        Force heading callback function. Calculates torque needed to achieve
            desired state.

        :param msg: contains commanded state information
        :type msg: VRXState Ros msg

        '''
        wrench_msg = Wrench()
        wrench_msg.force = msg.force
        self.psi_c = msg.heading
        # print(f'goal: {self.psi_c}, estimated: {self.psi_c}')

        self.add_or_subtract_pi()

        Tau = self.pid.update(np.array([[self.psi]]), self.psi_c)
        if abs(Tau) > self.tau_max:
            Tau = np.sign(Tau) * self.tau_max
        wrench_msg.torque = Vector3(0.0, 0.0, Tau)
        
        self.wrench_d_pub.publish(wrench_msg)

    def add_or_subtract_pi(self):
        psi_plus_2pi = self.psi + 2*np.pi
        psi_minus_2pi = self.psi - 2*np.pi
        diff_plus_2pi = abs(psi_plus_2pi - self.psi_c)
        diff_minus_2pi = abs(psi_minus_2pi - self.psi_c)
        diff_unchanged = abs(self.psi - self.psi_c)
        if np.min([diff_plus_2pi, diff_minus_2pi, diff_unchanged]) == diff_plus_2pi:
            self.psi = psi_plus_2pi
        elif np.min([diff_plus_2pi, diff_minus_2pi, diff_unchanged]) == diff_minus_2pi:
            self.psi = psi_minus_2pi
        return

if __name__ == '__main__':
    rospy.init_node('heading_control_node', anonymous=True)
    try:
        ros_node = HeadingControlNode()
        ros_node.run()
    except rospy.ROSInterruptException:
        pass
