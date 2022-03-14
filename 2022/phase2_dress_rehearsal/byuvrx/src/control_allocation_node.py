#!/usr/bin/env python3

# system imports
import numpy as np
from numpy.linalg import inv

# ROS imports
import rospy
from geometry_msgs.msg import Wrench
from std_msgs.msg import Float32
from vrx_gazebo.msg import Task

# local imports
# from control_allocation_optimization import wrench_diff_norm
#import tools.enable_table as ENABLE 
from tools import enable_table


class ControlAllocationNode:
    '''
    :class ControlAllocationNode: Interprets a thrust and torque into actuator outputs 
        using linear pseudoinverse matrix

    '''

    def __init__(self):
        '''
        initialization function
        '''
        # Initialize private variables
        self.throttle_right = 0.5
        self.angle_right = 0.0
        self.throttle_left = 0.5
        self.angle_left = 0.0

        # Thruster limits
        self.thrust_max = 250
        self.thrust_min = -100

        self.thruster_right = np.array(rospy.get_param(
            'control_allocation/thruster_right_pos'))
        self.thruster_left = np.array(rospy.get_param(
            'control_allocation/thruster_left_pos'))

        # Control Allocation Matirx - M
        self.M = np.array([[1, 0, 1, 0],
                          [0, 1, 0, 1],
                          [(-1*self.thruster_left[1]), self.thruster_left[0], (-1*self.thruster_right[1]), self.thruster_right[0]]])
        self.average_diff = np.array([[0, 0, 0]]).T
        self.num_callbacks = 0

        self.enabled = False
        self.setup = False
        self.task_info_sub = rospy.Subscriber("/vrx/task/info", Task, self.task_info_callback, queue_size=1)

    def task_info_callback(self, msg):
        if self.setup:
            return
        elif msg.name in enable_table.CONTROL_ALLOCATION_TABLE:
            self.enabled = True
            self.setup = True
            self.setup_pubs_subs()

    def setup_pubs_subs(self):

        # Initialize ROS
        self.thrust_right_pub = rospy.Publisher(
            "/wamv/thrusters/right_thrust_cmd", Float32, queue_size=1)
        self.angle_right_pub = rospy.Publisher(
            "/wamv/thrusters/right_thrust_angle", Float32, queue_size=1)
        self.thrust_left_pub = rospy.Publisher(
            "/wamv/thrusters/left_thrust_cmd", Float32, queue_size=1)
        self.angle_left_pub = rospy.Publisher(
            "/wamv/thrusters/left_thrust_angle", Float32, queue_size=1)

        rospy.Subscriber("/vrx_controller/wrench_d", Wrench, self.wrench_callback)

    def run(self):
        '''
        This loop runs until ROS node is killed
        '''
        rospy.spin()

    def wrench_callback(self, msg):
        '''
        wrench_d topic callback function. Calculates actuator outputs based on 
        desired thrusts and torques.

        :param msg: desired thrust and torque
        :type msg: Wrench Ros msg

        '''
        thrust_d = msg.force
        torque_d = msg.torque
        wrench_d = np.array([[thrust_d.x], [thrust_d.y], [torque_d.z]])
        # print(wrench_d)

        thrust_right, thrust_left, self.angle_right, self.angle_left \
            = self.compute_thruster_outputs(wrench_d)

        while (thrust_right > self.thrust_max or thrust_right < self.thrust_min) or \
            (thrust_left > self.thrust_max or thrust_left < self.thrust_min):
            wrench_d[0,0] *= .9
            wrench_d[1,0] *= .9
            wrench_d[2,0] *= .95
            # print(wrench_d)
            thrust_right, thrust_left, self.angle_right, self.angle_left \
                = self.compute_thruster_outputs(wrench_d)
        
        # compute achieved thrust/torque
        # print(np.array([[thrust_left*np.cos(self.angle_left).item(0)],
        #                                      [thrust_left*np.sin(self.angle_left).item(0)],
        #                                      [thrust_right*np.cos(self.angle_right).item(0)],
        #                                      [thrust_right*np.sin(self.angle_right).item(0)]]))
        wrench_achieved = self.M @ np.array([[(thrust_left*np.cos(self.angle_left)).item(0)],
                                             [(thrust_left*np.sin(self.angle_left)).item(0)],
                                             [(thrust_right*np.cos(self.angle_right)).item(0)],
                                             [(thrust_right*np.sin(self.angle_right)).item(0)]])
        wrench_diff = np.abs(wrench_achieved - wrench_d)
        self.num_callbacks += 1
        self.average_diff = (self.num_callbacks - 1) * (self.average_diff / self.num_callbacks) \
            + wrench_diff / self.num_callbacks        

        self.throttle_right = self.thrust_to_throttle(thrust_right)
        self.throttle_left = self.thrust_to_throttle(thrust_left)
        # print(f'thrust_left: {thrust_left}\nthrust_right: {thrust_right}\nangle_left: {self.angle_left}\nangle_right: {self.angle_right}')

        self.publish_actuator_cmds()

    def compute_thruster_outputs(self, wrench_d):
        '''

        Computes thruster outputs from a desired thrust and torque

        :param wrench_d: desired force and torque
        :type wrench_d: Numpy array

        :return: the thrust magnitude of the right thruster
        :return: angle of the right thruster
        :return: the thrust magnitude of the left thruster
        :return: angle of the left thruster

        '''
        rectangular_actuator_commands, _, _, _ = np.linalg.lstsq(
            self.M, wrench_d, rcond=None)
        thrust_left_x = rectangular_actuator_commands[0]
        thrust_left_y = rectangular_actuator_commands[1]
        thrust_right_x = rectangular_actuator_commands[2]
        thrust_right_y = rectangular_actuator_commands[3]

        thrust_left_total = np.sqrt(thrust_left_x**2+thrust_left_y**2)
        thrust_right_total = np.sqrt(thrust_right_x**2+thrust_right_y**2)

        angle_left = np.arctan2(thrust_left_y, thrust_left_x)
        angle_right = np.arctan2(thrust_right_y, thrust_right_x)

        angle_left, thrust_left_total = self.make_angle_point_forward(angle_left, thrust_left_total)
        angle_right, thrust_right_total = self.make_angle_point_forward(angle_right, thrust_right_total)
        
        return thrust_right_total, thrust_left_total, angle_right, angle_left

    def make_angle_point_forward(self, angle_orig, thrust_orig):
        '''

        WAM-V can only have forward pointing angles (-pi/2 < angle < pi/2).
        Fixes a negative angle to be a negative thrust with a forward pointing angle.

        :param angle_orig: original angle
        :type angle_orig: float
        :param thrust_orig: original thrust value
        :type thrust_orig: float

        :return: new angle
        :return: new thrust

        '''
        if np.absolute(angle_orig) > np.pi/2:
            thrust = -thrust_orig
            if angle_orig > 0:
                angle = angle_orig - np.pi
            elif angle_orig < 0:
                angle = angle_orig + np.pi
        else:
            angle = angle_orig
            thrust = thrust_orig
        return angle, thrust

    def thrust_to_throttle(self, thrust):
        '''

        Calculates the throttle mapping for a desired thrust

        :param thrust: desired thrust output
        :type thrust: float
        :return: throttle from -1.0 to 1.0
        :rtype: float

        '''
        # set thrust maximum
        if thrust > 250:
            thrust = 250

        # set thrust minimum
        elif thrust < -100:
            thrust = -100

        # if thrust is reverse then set these varaible
        if thrust > 0:
            a = -.01
            k = 59.82
            b = 5.0
            v = 0.38
            c = 0.56
            m = 0.28

        # if thrust is forward then set these varaible
        elif thrust < 0:
            a = -199.13
            k = -.09
            b = 8.84
            v = 5.34
            c = 0.99
            m = -.57

        else:
            return 0

        # determine throttle from thrust
        x = (np.log(((k-a)/(thrust-a))**v-c))/(-b)+m
        return x

    def publish_actuator_cmds(self):
        '''
        Publishes desired thruter angle and throttles
        '''
        msg_thrust_r = Float32(self.throttle_right)
        msg_thrust_l = Float32(self.throttle_left)
        msg_angle_r = Float32(self.angle_right)
        msg_angle_l = Float32(self.angle_left)

        self.thrust_right_pub.publish(msg_thrust_r)
        self.thrust_left_pub.publish(msg_thrust_l)
        self.angle_right_pub.publish(msg_angle_r)
        self.angle_left_pub.publish(msg_angle_l)


if __name__ == '__main__':
    rospy.init_node('control_allocation_node', anonymous=True)
    try:
        ros_node = ControlAllocationNode()
        ros_node.run()
    except rospy.ROSInterruptException:
        pass
