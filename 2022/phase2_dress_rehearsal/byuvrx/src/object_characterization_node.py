#!/usr/bin/env python3

import cv2
import numpy as np
import rospy
import sys 
import os
import threading
import logging
import signal 

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from vrx_gazebo.msg import Task

from robowalrus.msg import BoundingBox
#import tools.enable_table as ENABLE
from tools import enable_table

class middle_camera:

    def __init__(self):
        self._lock = threading.Lock()
        self.image_data = None
        self.run = True

        dir = os.path.dirname(os.path.realpath(__file__))
        self.net = cv2.dnn.readNet(dir + '/yolov3_training_last.weights', dir + '/yolov3_testing.cfg')

        self.font = cv2.FONT_HERSHEY_PLAIN
        self.colors = np.random.uniform(0, 255, size=(100, 3))
        
        self.classes = []
        with open(dir + "/classes.txt", "r") as f:
            self.classes = f.read().splitlines()

        self.enabled = False
        self.setup = False
        self.task_info_sub = rospy.Subscriber("/vrx/task/info", Task, self.task_info_callback, queue_size=1)

    def task_info_callback(self, msg):
        if self.setup:
            return
        elif msg.name in enable_table.OBJECT_CHARACTERIZATION:
            self.enabled = True
            self.setup = True
            self.setup_pubs_subs()

    def setup_pubs_subs(self):
        self.image_sub = rospy.Subscriber("/wamv/sensors/cameras/middle_camera/image_raw", Image, self.callback, queue_size=1)
        self.box_pub = rospy.Publisher("/vrx_controller/BoundingBox", BoundingBox, queue_size=1)

    #Currently this code runs really slow. Perhaps it would better to not run all in the callback
    #Needs to be optimized in some way.
    #Watch the video https://www.youtube.com/watch?v=1LCb1PVqzeY to understand more of the code
    #Could give hints of how to optimize
    def callback(self, data):
        with self._lock:
            self.image_data = data
        
    def image_processing(self):
        while self.run:
            if not self.enabled:
                continue
            if self.image_data is not None:
                with self._lock:
                    data = self.image_data
                    self.image_data = None

                bridge = CvBridge()
                try:
                    cv_image = bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
                except CvBridgeError as e:
                    rospy.logerr(e)

                img = cv_image
                height, width, _ = img.shape

                blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
                self.net.setInput(blob)
                output_layers_names = self.net.getUnconnectedOutLayersNames()
                layerOutputs = self.net.forward(output_layers_names)

                boxes = []
                confidences = []
                class_ids = []

                #This seems to be detecting the objects with bounding box confidence and class.
                for output in layerOutputs:
                    for detection in output:
                        scores = detection[5:] #this means from the 6th element onward... do we want this???
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if confidence > 0.3:
                            center_x = int(detection[0]*width)
                            center_y = int(detection[1]*height)
                            w = int(detection[2]*width)
                            h = int(detection[3]*height)

                            x = int(center_x - w/2)
                            y = int(center_y - h/2)
                            boxes.append([x, y, w, h])
                            confidences.append((float(confidence)))
                            class_ids.append(class_id)

                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)
                #This seems to be doing the boxes
                if len(indexes)>0:
                    for i in indexes.flatten():
                        x, y, w, h = boxes[i]
                        label = str(self.classes[class_ids[i]])
                        confidence = str(round(confidences[i],2))
                        color = self.colors[i]
                        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
                        cv2.putText(img, label + " " + confidence, (x, y+20), self.font, 2, (255,255,255), 2)
                        #Make and publish the bounding box
                        bounding_box = BoundingBox()
                        bounding_box.x = x
                        bounding_box.y = y
                        bounding_box.w = w
                        bounding_box.h = h
                        bounding_box.object_id = label
                        self.box_pub.publish(bounding_box)
                        print(label, rospy.Time.now().secs)

                #cv2.imshow('Image', img)
                #key = cv2.waitKey(1)
                #if key==27:
                #   break



def main():
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,  datefmt="%H:%M:%S")

    #More cameras could be added
    node = middle_camera()
    
    def handler(*args):
        node.run = False
        rospy.loginfo("Shutting down")
        x.join()
        logging.info('Main: Closing threads')
        cv2.destroyAllWindows()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, handler)
    
    try:
        logging.info('Main: Initialize middle camera node')

        x = threading.Thread(target=node.image_processing)

        logging.info('Main: Starting image processing thread')

        x.start()

        logging.info('Main: Calling rospy spin')

        rospy.spin()
    except KeyboardInterrupt: # this doesn't actually catch anything
        node.run = False
        rospy.loginfo("Shutting down")
        x.join()
        logging.info('Main: Closing threads')
        cv2.destroyAllWindows()
        sys.exit(0)
    
 

if __name__ == '__main__':     
    rospy.init_node('object_charcterization_node', anonymous=True)
    main()


#Below is a good camera test. You can swap out 
# the subscribed topic to switch cameras. 

#!/usr/bin/env python

# import rospy
# import cv2
# import time
# import sys

# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge, CvBridgeError

# class front_left_camera:

#   def __init__(self):
#     self.image_sub = rospy.Subscriber("/wamv/sensors/cameras/front_left_camera/image_raw", Image, self.callback)

#   def callback(self,data):
#     bridge = CvBridge()

#     try:
#       cv_image = bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
#     except CvBridgeError as e:
#       rospy.logerr(e)
    
#     image = cv_image

#     h = data.height
#     w= data.width

#     resized_image = cv2.resize(image, (w, h)) 

#     #cv2.imshow("Camera output normal", image)
#     cv2.imshow("Camera FL output resized", resized_image)

#     cv2.waitKey(3)

# def main():
#   front_left_camera()
#   time.sleep(1)
#   #front_right_camera()
#   #time.sleep(1)
#   #middle_right_camera()
#   #time.sleep(1)
#   #signal.signal(signal.SIGINT, keyboardInterruptHandler)

#   try:
#     rospy.spin()
#   except KeyboardInterrupt:
#     rospy.loginfo("Shutting down")
  
#   cv2.destroyAllWindows()
#   sys.exit(0)

# if __name__ == '__main__':
#     rospy.init_node('object_characterization_node', anonymous=True)
#     main()
