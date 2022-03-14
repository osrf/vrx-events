#!/usr/bin/env python

#This program is useful for aquiring images for training. Run this node and place objects in the camera's view. Then 
#screenshot


import rospy
import cv2
import time
import sys

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class front_left_camera:

  def __init__(self):
    self.image_sub = rospy.Subscriber("/wamv/sensors/cameras/front_left_camera/image_raw", Image, self.callback)

  def callback(self,data):
    bridge = CvBridge()

    try:
      cv_image = bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
    except CvBridgeError as e:
      rospy.logerr(e)
    
    image = cv_image

    h = data.height
    w= data.width

    resized_image = cv2.resize(image, (w, h)) 

    #cv2.imshow("Camera output normal", image)
    cv2.imshow("Camera FL output resized", resized_image)

    cv2.waitKey(3)

class middle_right_camera:

  def __init__(self):
    self.image_sub = rospy.Subscriber("/wamv/sensors/cameras/middle_right_camera/image_raw", Image, self.callback)

  def callback(self,data):
    bridge = CvBridge()

    try:
      cv_image = bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
    except CvBridgeError as e:
      rospy.logerr(e)
    
    image = cv_image

    h = data.height
    w= data.width

    resized_image = cv2.resize(image, (w, h)) 

    #cv2.imshow("Camera output normal", image)
    cv2.imshow("Camera MR output resized", resized_image)

    cv2.waitKey(3)


def main():
  front_left_camera()
  time.sleep(1)
  #front_right_camera()
  #time.sleep(1)
  #middle_right_camera()
  #time.sleep(1)
  #signal.signal(signal.SIGINT, keyboardInterruptHandler)

  try:
    rospy.spin()
  except KeyboardInterrupt:
    rospy.loginfo("Shutting down")
  
  cv2.destroyAllWindows()
  sys.exit(0)

if __name__ == '__main__':
    rospy.init_node('camera_test', anonymous=True)
    main()