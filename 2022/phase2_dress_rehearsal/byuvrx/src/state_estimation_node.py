#!/usr/bin/env python3

# system imports
import numpy as np
import inekf
import math
import time

# ROS imports
import rospy
from geometry_msgs.msg import Vector3Stamped
from sensor_msgs.msg import NavSatFix
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from std_msgs.msg import Header
from vrx_gazebo.msg import Task

# Inekf Imports
from inekf import InertialProcess
from inekf import SE3, InEKF, ERROR

# local imports
from robowalrus.msg import VRXState
from tools import enable_table

class StateEstimationNode:
    """

    :class StateEstimationNode: Take in GPS and IMU data, fuses them, and outputs position
        of WAMV and its derivative, and yaw with its derivative. Uses InEKF package to estimate the data,
        rather than a generic EFK package.

    """

    def __init__(self):
        """
        Initialization function. Sets up subscribers to IMU and GPS sensors. Calls the init sensors function.

        """

        # Variables used to maintaining correct ordering of sensors/published messages
        self.first_gps_reading = True
        self.first_imu_reading = True
        self.first_pose_reading = True
        self.init_sensors()

        # parameters
        self.globe_radius = rospy.get_param('state_estimation/globe_radius')

        # Creat self variables and initialize values
        self.prev_psi_reading = 0.0
        self.init_lat = 0.0
        self.init_long = 0.0
        self.t = 0.0
        self.velocity = np.array([0, 0, 0])
        self.psi = 0.0
        self.psi_d1 = None
        self.gps_sensor_offset = 0.3
        self.position = np.array([0.0, 0.0, 0.0])
        self.angular_velocity = np.array([0, 0, 0])
        self.true_state_phi = 0.0
        self.true_state_theta = 0.0

        self.desired_state = VRXState()

        self.last_imu_time_stamp = 0.0

        self.sigma = 1/15
        Ts = 1/15
        self.beta = (2.0*self.sigma-1/15)/(2.0*self.sigma+1/15)

        self.enabled = False
        self.setup = False
        self.task_info_sub = rospy.Subscriber("/vrx/task/info", Task, self.task_info_callback, queue_size=1)

    def task_info_callback(self, msg):
        if self.setup:
            return
        elif msg.name in enable_table.STATE_ESTIMATION:
            self.enabled = True
            self.setup = True
            self.setup_pubs_subs()

    def setup_pubs_subs(self):
        # Initialize ROS
        self.state_pub = rospy.Publisher("/vrx_controller/estimated_state", VRXState, queue_size=1)
        self.state_msg = VRXState()
        self.state_msg.psi = 0.0
        self.true_state_msg = VRXState()
        self.point = Point()
        self.pose_pub = rospy.Publisher("/pose", VRXState, queue_size=1)
        self.offset_pub = rospy.Publisher("/vrx_controller/lateral_offset", Point, queue_size=1)

        rospy.Subscriber("wamv/sensors/imu/imu/data", Imu, self.imu_callback)
        rospy.Subscriber("wamv/sensors/gps/gps/fix",NavSatFix, self.fix_callback)
        rospy.Subscriber("wamv/sensors/gps/gps/fix_velocity", Vector3Stamped,self.fix_velocity_callback)
        rospy.Subscriber("wamv/sensors/position/p3d_wamv", Odometry, self.true_state_callback)
        rospy.Subscriber("/vrx_controller/trajectory", VRXState, self.desired_state_callback)

    def true_state_callback(self, data):
        """
        Used for reading true state data when the wamv_p3d component is included in component_config.yaml.

        :param data: The data coming in from the wamv/sensors/position/p3d_wamv topic

        """
        
        orientation = data.pose.pose.orientation
        t0 = +2.0 * (orientation.w * orientation.z +
                     orientation.x * orientation.y)
        t1 = +1.0 - 2.0 * (orientation.y * orientation.y +
                               orientation.z * orientation.z)
        self.psi = math.atan2(t0, t1)
        self.true_state_phi = math.atan2(2*(orientation.w*orientation.x + orientation.y*orientation.z),
            1-2*(orientation.x**2 + orientation.y**2))
        self.true_state_theta = math.asin(2*(orientation.w*orientation.y - orientation.x*orientation.z))

        self.angular_velocity = np.array([data.twist.twist.angular.x, data.twist.twist.angular.y])
        self.velocity = np.array([data.twist.twist.linear.x, data.twist.twist.linear.y, data.twist.twist.linear.z])
        self.position = np.array([data.pose.pose.position.x, data.pose.pose.position.y, data.pose.pose.position.z])
        # self.position += np.array([np.cos(self.psi)])

        if self.first_pose_reading:
            self.init_true_position = [self.position[0] + self.gps_sensor_offset * np.cos(self.psi), self.position[1] + self.gps_sensor_offset * np.sin(self.psi), self.position[2]]
            self.init_true_psi = self.psi
            self.first_pose_reading = False

        self.position -= self.init_true_position

    def desired_state_callback(self, data):
        self.desired_state = data

    def imu_callback(self, data):
        """

        Take the subscribed data and update the Kalman filter. Extract yaw from the State matrix

        :param data: The data coming from the wamv/sensors/imu/imu/data topic.

        """
        # rospy.loginfo(rospy.get_caller_id() + "\nImu \n Orientation: %s\n Ang. Vel: %s \n Lin. Acc. %s\n",
        # data.orientation, data.angular_velocity, data.linear_acceleration)
        timestamp = float(data.header.stamp.secs + data.header.stamp.nsecs * 1e-9)
        # This formula finds yaw from a quaternion
        if (self.first_imu_reading):
            t0 = +2.0 * (data.orientation.w * data.orientation.z +
                         data.orientation.x * data.orientation.y)
            t1 = +1.0 - 2.0 * (data.orientation.y * data.orientation.y +
                               data.orientation.z * data.orientation.z)
            self.xi[2] = math.atan2(t0, t1)
            # later when we call Predict(), timestamp - self.last_imu_timestamp will equal 1/15;
            # not sure why, but the IMU callback seems to be called at 15 Hz
            self.last_imu_timestamp = timestamp - 1/15
            self.first_imu_reading = False

        t0 = +2.0 * (data.orientation.w * data.orientation.z +
                         data.orientation.x * data.orientation.y)
        t1 = +1.0 - 2.0 * (data.orientation.y * data.orientation.y +
                               data.orientation.z * data.orientation.z)
        yaw_reading = math.atan2(t0, t1)

        imu_data = np.zeros(6)

        imu_data[0] = data.angular_velocity.x
        imu_data[1] = data.angular_velocity.y
        imu_data[2] = data.angular_velocity.z
        imu_data[3] = data.linear_acceleration.x
        imu_data[4] = data.linear_acceleration.y
        imu_data[5] = data.linear_acceleration.z

        # # the second argument is time since last measurement taken
        time_diff = timestamp - self.last_imu_timestamp
        self.state = self.iekf.Predict(imu_data, time_diff)
        self.last_imu_timestamp = float(timestamp)

        # eventually publish yaw
        self.state_msg.psi = yaw_reading
        self.state_msg.x = self.state.State[0, 4] - self.gps_sensor_offset*np.cos(self.state_msg.psi)
        self.state_msg.y = self.state.State[1, 4] - self.gps_sensor_offset*np.sin(self.state_msg.psi)

        # Dirty derivative
        if self.psi_d1 is not None:
            # accounts for wrapparound of yaw
            if abs(self.state_msg.psi - self.psi_d1) > abs(self.state_msg.psi - (self.psi_d1 + 2*np.pi)):
                self.psi_d1 += 2*np.pi
            elif abs(self.state_msg.psi - self.psi_d1) > abs(self.state_msg.psi - (self.psi_d1 - 2*np.pi)):
                self.psi_d1 -= 2*np.pi
            self.state_msg.psidot = self.beta * self.state_msg.psidot \
                             + (1 - self.beta) * ((self.state_msg.psi - self.psi_d1) / (1/15))
        else:
            self.state_msg.psidot = 0
        self.prev_psi_reading = yaw_reading
        self.state_msg.xdot = self.state.State[0, 3] + self.state_msg.psidot*self.gps_sensor_offset*np.sin(self.state_msg.psi)
        self.state_msg.ydot = self.state.State[1, 3] - self.state_msg.psidot*self.gps_sensor_offset*np.cos(self.state_msg.psi)
        self.state_pub.publish(self.state_msg)
        if not self.first_gps_reading:
            self.offset_pub.publish(self.point)

        self.true_state_msg.xdot = self.velocity[0]
        self.true_state_msg.ydot = self.velocity[1]
        
        self.true_state_msg.x = self.position[0]
        self.true_state_msg.y = self.position[1]
        self.true_state_msg.psi = self.psi
        self.true_state_msg.psidot = 0.0   
        self.psi_d1 = self.state_msg.psi     
        # self.pose_pub.publish(self.true_state_msg)
        # print(f'x: {self.state_msg.x}, y: {self.state_msg.y}')
        #print(np.round(self.state.State, 4))
        if True:
            print(f'{self.t}:')
            print(f'  estimated_state:')
            print(f'     linear_velocity: {[self.state_msg.xdot, self.state_msg.ydot, 0.0]}')
            print(f'     position: {[self.state_msg.x, self.state_msg.y, 0.0]}')
            #print(f'     angular_velocity:')
            #print(f'     linear_acceleration:')
            print(f'     orientation: {[0.0, 0.0, yaw_reading]}')
            print(f'  true_state:')
            print(f'     linear_velocity: {[self.true_state_msg.xdot, self.true_state_msg.ydot, self.velocity[2]]}')
            print(f'     angular_velocity: {[self.angular_velocity[0], self.angular_velocity[1], 0.0]}')
            print(f'     position: {[self.position[0], self.position[1], self.position[2]]}')
            #print(f'     linear acceleration: {[]}')
            print(f'     orientation: {[self.true_state_phi, self.true_state_theta, self.true_state_msg.psi]}')
            print(f'  desired_state:')
            print(f'     position: {[self.desired_state.x, self.desired_state.y, 0.0]}')
            print(f'     orientation: {[0.0, 0.0, self.desired_state.psi]}')
        # print(self.state.State)
        self.t += time_diff
        # print(float(data.header.stamp.nsecs * pow(10,-9) + data.header.stamp.secs))

    def fix_velocity_callback(self, data):
        """

        Take the subscribed data and set member variables of the class relating to the fix velocity.

        :param data: The data coming from the wamv/sensors/gps/gps/fix_velocity topic.

        """
        # rospy.loginfo(rospy.get_caller_id() + "\nVelocity vector: %s\n Header: %s\n Stamp: %s",
        #               data.vector, data.header, data.header.stamp)

        inertial_vel = \
            np.array([[np.cos(np.pi/2), -np.sin(np.pi/2), 0.0],
                      [np.sin(np.pi/2), np.cos(np.pi/2), 0.0],
                      [0.0, 0.0, 1.0]]) @ \
            np.array([[data.vector.x], [data.vector.y], [data.vector.z]])

        gps_vel_data = np.zeros(3)
        gps_vel_data[0] = inertial_vel.item(0)
        gps_vel_data[1] = inertial_vel.item(1)
        gps_vel_data[2] = 0.0
        self.iekf.Update(gps_vel_data, 'gps_vel')

    def fix_callback(self, data):
        """

        Take the subscribed gps data and convert to cartesian coordiantes. Update the Kalman filter with these new readings

        :param data: The data coming from the wamv/sensors/gps/gps/fix topic.

        """
        # rospy.loginfo(rospy.get_caller_id() + "\nFix: \nLat: %s\n Long: %s\n Alt: %s",
        # data.latitude, data.longitude, data.altitude)

        # set xi for inekf if this is the first reading
        # TODO How do we make this "0, 0" ...right now we are keeping track of the offset and subtracting from every
        # subsequent reading

        # doesn't run until imu has run
        if (self.first_imu_reading):
            return

        if (self.first_gps_reading):
            self.set_init_lat_long(data.latitude, data.longitude)
            self.xi[8] = float(data.altitude)
            self.first_gps_reading = False

        gps_x, gps_y = self.convert_to_cartesian(data.latitude, data.longitude)
        gps_data = np.array([gps_x, gps_y, 0.0])
        self.iekf.Update(gps_data, 'gps')
        # print(self.state.State.item())

    def set_init_lat_long(self, lat, long):
        self.init_lat = float(lat)*np.pi/180
        self.init_long = float(long)*np.pi/180
        x_offset = self.gps_sensor_offset*np.cos(self.state_msg.psi) / self.globe_radius
        y_offset = self.gps_sensor_offset*np.sin(self.state_msg.psi) / self.globe_radius
        # self.init_lat += x_offset
        # self.init_long += y_offset
        self.point.x = self.init_long #+ x_offset
        self.point.y = self.init_lat #+ y_offset
        # self.offset_pub.publish(self.point)

    def convert_to_cartesian(self, lat, long):
        """
        This converts the given lat, long, and alt parameters to cartesian coordinates (x, y, z)
        In frame ENU.

        :param lat: Latitude given from GPS
        :param long: Longitude given from GPS
        :param alt: Altitude given from GPS

        :return: x, y, and z coordinates

        """
        lat_rad = lat*np.pi/180
        long_rad = long*np.pi/180
        std_parallel = self.init_lat
        x = self.globe_radius * (long_rad - self.init_long) * np.cos(std_parallel)
        y = self.globe_radius * (lat_rad - self.init_lat)
        return x, y

    def run(self):
        """
        This loop runs until ROS node is killed. Prevents callbacks from being called concurrently
        """
        rospy.spin()

    def init_sensors(self):
        """
        :init_sensors: This initializes the iekf, sets noise parameters
        """
        self.xi = np.zeros(15)

        b = np.array(rospy.get_param('state_estimation/inekf/gps_position/b'))
        # standard deviations
        sigma = rospy.get_param('state_estimation/inekf/gps_position/sigma')
        n = np.array([pow(sigma[0], 2), pow(sigma[1], 2), pow(sigma[2], 2)])
        noise = np.diag(n)
        self.gps = inekf.GenericMeasureModel[inekf.SE3[2, 6]](
            b, noise, inekf.ERROR.LEFT)

        b = np.array(rospy.get_param('state_estimation/inekf/gps_velocity/b'))
        sigma = rospy.get_param('state_estimation/inekf/gps_velocity/sigma')
        n = np.array([pow(sigma[0], 2), pow(sigma[1], 2), pow(sigma[2], 2)])
        noise = np.diag(n)
        self.gps_vel = inekf.GenericMeasureModel[inekf.SE3[2, 6]](
            b, noise, inekf.ERROR.LEFT)

        # TODO use our estimated initial data
        s = rospy.get_param('state_estimation/inekf/s')

        # we are using all zeros as the starting state xi
        # because we are treating our first reading as 0
        # s is our noise for gps
        x0 = SE3[2, 6](self.xi, np.diag(s))

        self.iekf = InEKF[InertialProcess](x0, ERROR.LEFT)
        self.iekf.addMeasureModel('gps', self.gps)
        self.iekf.addMeasureModel('gps_vel', self.gps_vel)

        gravity = rospy.get_param('state_estimation/gravity')
        self.iekf.pModel.setAccelBiasNoise(
            rospy.get_param('state_estimation/inekf/accelerometer_bias_noise') / gravity)
        self.iekf.pModel.setAccelNoise(
            rospy.get_param('state_estimation/inekf/accelerometer_noise') / gravity)
        self.iekf.pModel.setGyroBiasNoise(rospy.get_param('state_estimation/inekf/gyro_bias_noise'))
        self.iekf.pModel.setGyroNoise(rospy.get_param('state_estimation/inekf/gyro_noise'))

        # dt was delta time, time since last measurement taken
        # 6 numbers in data were vel and acc, no orientation.
        # IMU.predict
        # iekf.Update for other measurements
        # convert gps to x and y, add 0 for z
        # state.State = 5x5 matrix
        # Start with IMU and predict, easier to debug

if __name__ == '__main__':
    rospy.init_node('state_estimation_node', anonymous=True)
    try:
        ros_node = StateEstimationNode()
        ros_node.run()
    except rospy.ROSInterruptException:
        pass