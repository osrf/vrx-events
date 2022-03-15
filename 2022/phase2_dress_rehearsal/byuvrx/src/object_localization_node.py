#!/usr/bin/python3

from concurrent.futures import thread
import logging
import numpy as np
import rospy
import ros_numpy
import sys
import threading
import signal

from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Point
from geographic_msgs.msg import GeoPoseStamped
from vrx_gazebo.msg import Task

from robowalrus.msg import BoundingBox
from bounding_box import BoundingBoxClass
from geographic_msgs.msg import GeoPoseStamped
from State import State
from robowalrus.msg import VRXState
#import tools.enable_table as ENABLE
from tools import enable_table

RESIZE = 2 #Double the bounding box size so we can for sure see the lidar points

class lidar:
    
    def __init__(self):        
        
        # used for throwing out points because there are too many to process
        self.z_index = 2
        self.y_index = 1
        self.x_index = 0
        self.max_y = 1
        self.min_y = -10
        self.max_dist = 40 #throw out all points more than 40 meters out
        self.min_dist = 0
        self.index_modulo = 1 # keep one out of every x points
        self.k_param = np.array([[762.7249337622711, 0.0, 640.5], [ 0.0, 762.7249337622711, 360.5],[ 0.0, 0.0, 1.0]])
        self.P = np.array([[762.7249337622711, 0.0, 640.5, -53.39074536335898], [0.0, 762.7249337622711, 360.5, 0.0], [0.0, 0.0, 1.0, 0.0]])
        self.lidar_pitch = np.radians(15)
        self.lidar_offset = np.array([[.75, 0, 1.6]]).T
        self.globe_radius = 6370.0*pow(10, 3)

        self.has_offset = False
        self.bounding_boxes = list()
        self.lidar_data = None
        self._lock = threading.Lock()
        self.run = True

        self.state = State(rospy.get_param('trajectory_tracker/init_state/x_init'), 
            rospy.get_param('trajectory_tracker/init_state/y_init'),
            rospy.get_param('trajectory_tracker/init_state/psi_init'))
        self.gps_msg = GeoPoseStamped()

        self.enabled = False
        self.setup = False
        self.task_info_sub = rospy.Subscriber("/vrx/task/info", Task, self.task_info_callback, queue_size=1)

    def task_info_callback(self, msg):
        if self.setup:
            return
        elif msg.name in enable_table.OBJECT_LOCALIZATION:
            self.enabled = True
            self.setup = True
            self.setup_pubs_subs()

    def setup_pubs_subs(self):
        rospy.Subscriber("/wamv/sensors/lidars/lidar_wamv/points", PointCloud2, self.lidar_callback)
        rospy.Subscriber("/vrx_controller/BoundingBox", BoundingBox, self.bounding_box_callback)
        rospy.Subscriber("/vrx_controller/estimated_state", VRXState, self.state_callback)
        rospy.Subscriber("/vrx_controller/lateral_offset", Point, self.offset_callback)
        self.gps_pub = rospy.Publisher("/vrx/perception/landmark", GeoPoseStamped, queue_size=1)

    def state_callback(self, msg):
        '''
        state_estimation topic callback function. Updates global state variables

        :param msg: contains state information
        :type msg: VRXState Ros msg

        '''
        self.state = State(msg.x, msg.y, msg.psi)

    def offset_callback(self, data):
        '''
        offset topic callback function. Updates gloabl offset variables and formulates 
            trajectory msg if it has goal information already

        :param data: contains x and y offset information
        :type data: Point ROS msg

        '''
        if self.has_offset:
            return
        self.init_lat = data.y
        self.init_long = data.x
        self.has_offset = True

    # This callback copies the data to a variable for lidar processing. 
    # This is so we can process in 2 threads. The lock is important
    def lidar_callback(self, data):
        with self._lock:
            self.lidar_data = data

    # Wrapper function to run continually, waiting for the lidar_data to come
    # so they can process it
    def lidar_processing(self):
        #This loop will end when Ctrl-C is hit (see main)
        while self.run:
            if not self.enabled:
                continue
            if self.lidar_data is not None:
                if len(self.bounding_boxes) == 0:
                    with self._lock:
                        self.lidar_data = None
                else:
                    self.do_lidar_process()
    
    def do_lidar_process(self):
        # Make a local copy of the lidar data
        with self._lock:
            data = self.lidar_data
            self.lidar_data = None

        # x, y, z = forward, left, up
        xyz_array_body = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(data)
        R_lidar2camera = np.array([[0, -1, 0],
                                [0, 0, -1],
                                [1, 0, 0]])
        xyz_array = (R_lidar2camera @ np.array(xyz_array_body).T).T.tolist()

        # filter points 
        self.filter_points(xyz_array)

        # Resize all of the bounding boxes we received from Object Characterization message
        # Could be resized bigger or smaller, depending on the parameter RESIZE
        for box in self.bounding_boxes:
            box.resize(RESIZE)

        # Add all of the lidar points 
        for point in xyz_array:
            curr_pixel = self.calculate_pixel(point)
            for box in self.bounding_boxes:
                #This will only add if in the bounding box
                box.add_point(curr_pixel, point)

        i = 0
        while i < len(self.bounding_boxes):
            print(self.bounding_boxes[0].id)
            avg = self.bounding_boxes[0].average_lidar_points()
            print("Avg: ", avg)
            if avg is not None:
                gps_long, gps_lat = self.lidar2gps(avg)
                gps_msg = GeoPoseStamped()
                gps_msg.header.frame_id = self.bounding_boxes[0].id
                if gps_msg.header.frame_id == "turtle":
                    gps_msg.header.frame_id = "mb_round_buoy_orange"
                gps_msg.pose.position.latitude = gps_lat
                gps_msg.pose.position.longitude = gps_long
                self.gps_pub.publish(gps_msg)
                print(gps_long)
                print(gps_lat)
            del self.bounding_boxes[i]
            i += 1
            
            #Publish point here?
            #gps_msg.header.frame_id = 
            #gps_msg.pose.position.latitude = 
            #gps_msg.pose.position.longitude = 


    def bounding_box_callback(self, data):
        new_box = BoundingBoxClass(x=data.x, y=data.y, width=data.w, height=data.h, id=data.object_id)
        for i in range(len(self.bounding_boxes)):

            # The boundin box object, defined in bounding_box.py, defines equality as
            # having the same name and being *close* to the same bounding box area
            # Delete duplicate boxes and replace with the new one
            if self.bounding_boxes[i] == new_box:
                del self.bounding_boxes[i]
                break
        self.bounding_boxes.append(new_box)           

    # Because there are so many lidar points, to speed up processing we throw out any points behind
    # us (or even outside of the camera's POV). We also just throw out certain points, since we only 
    # need a few to get an idea where the object is. 
    def filter_points(self, xyz_array):
        # keep one out of every self.index_modulo points
        i = len(xyz_array) - 1
        while i > 0:
            if i % self.index_modulo != 0:
                xyz_array.pop(i)
            i -= 1
        # throw out points that are behind the vehicle or too high or too low or too far away
        i = len(xyz_array) - 1
        while i > 0:
            dist = np.sqrt(pow(xyz_array[i][self.x_index], 2) + pow(xyz_array[i][self.y_index], 2) + pow(xyz_array[i][self.z_index], 2))
            if xyz_array[i][self.z_index] < 0 or xyz_array[i][self.y_index] > self.max_y or xyz_array[i][self.y_index] < self.min_y \
                or dist > self.max_dist or dist < self.min_dist:
                xyz_array.pop(i)
            i -= 1

    # This is a complicated function that maps a lidar point to a pixel seen on the camera.
    # See https://en.wikipedia.org/wiki/Pinhole_camera_model
    # Also, ask Dr. Mangelson about his slides he has on the subject. That's where the 
    # equations came from. A lot of the variables came from the camera info topic
    def calculate_pixel(self, coord):           
        coord = np.concatenate((np.array([coord]).T, [[1]]), axis=0)
        augmented_identity = np.eye(3,4)
        x_prime = self.k_param @ augmented_identity @ coord
        x_prime[0:2, :] /= x_prime[2, 0]
        return x_prime[0:2, :]

    # p_o_in_ell: position of the object with respect to the lidar in the lidar frame
    # Converts from lidars frame to gps's frame so we can publish the coordinates
    def lidar2gps(self, p_o_in_ell):
        # rotation from lidar frame to body frame
        R_ell2b = np.array([[np.cos(self.lidar_pitch), 0, np.sin(self.lidar_pitch)],
                           [0, 1, 0],
                           [-np.sin(self.lidar_pitch), 0, np.cos(self.lidar_pitch)]]) \
            @ np.array([[0, 0, 1],
                       [-1, 0, 0],
                       [0, -1, 0]])
        # rotation from body frame to inertial frame
        R_b2i = np.array([[np.cos(self.state.psi), -np.sin(self.state.psi), 0],
                          [np.sin(self.state.psi), np.cos(self.state.psi), 0],
                          [0, 0, 1]])
        # object position in body frame
        p_o_in_b = R_ell2b @ p_o_in_ell.reshape((3,1)) + self.lidar_offset
        # object position in inertial frame
        p_o_in_i = R_b2i @ p_o_in_b + self.state.position
        
        print(p_o_in_b)
        print(p_o_in_i)

        obj_long = np.degrees(p_o_in_i.item(0) / (self.globe_radius*np.cos(self.init_lat)) \
            + self.init_long)
        obj_lat = np.degrees(p_o_in_i.item(1) / self.globe_radius + self.init_lat)

        return obj_long, obj_lat
    
def main():
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.DEBUG,  datefmt="%H:%M:%S")
    
    # This will catch Ctrl-C so you can exit
    def handler(*args):
        node.run = False # Break out of the whileloop in lidar_processing
        rospy.loginfo("Shutting down")
        second_thread.join() # Make sure to join the threads
        sys.exit(0)
    
    signal.signal(signal.SIGINT, handler)

    try:
        node = lidar()
        second_thread = threading.Thread(target=node.lidar_processing)
        second_thread.start()
        rospy.spin()
    except KeyboardInterrupt: # this doesn't actually catch anything...
        node.run = False
        #rospy.loginfo("Shutting down")
        second_thread.join()
        sys.exit(0)


if __name__ == '__main__':
    rospy.init_node('object_localization_node', anonymous=True)
    main()






