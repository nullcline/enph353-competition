#!/usr/bin/env python

import rospy
import cv2
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
from std_msgs.msg import Time
import copy

import tensorflow as tf 
import numpy as np

from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model

from plate_reader import PlateReader
from imitator import Imitator

# Required setup for running the models in image callback
sess = tf.Session()
graph = tf.get_default_graph()
set_session(sess)
outerloop_model     = load_model('/home/andrew/ros_ws/src/2020T1_competition/controller/models/OLv1.h5')
intersection_model  = load_model('/home/andrew/ros_ws/src/2020T1_competition/controller/models/Xv1.h5')
innerloop_model     = load_model('/home/andrew/ros_ws/src/2020T1_competition/controller/models/ILv0.h5')
license_plate_model = load_model('/home/andrew/ros_ws/src/2020T1_competition/controller/models/plate_number_model.h5')
id_plate_model      = load_model('/home/andrew/ros_ws/src/2020T1_competition/controller/models/plate_id_model.h5')

class Controller:

  def __init__(self):

  
    self.init_time = rospy.get_time()
    self.plates = rospy.Publisher('/license_plate', String, queue_size=1)
    rospy.sleep(1)

    # publisher for movement
    self.pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)
    self.move = Twist()
    self.move_delay = 0
    self.x = 0.5
    self.z = 2
    self.theta = 0
    self.erosion_thresh = 100
    self.truck_thresh = 50
    
    # image stuff
    self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image, self.callback, queue_size=1, buff_size=2**24) 
    self.bridge = CvBridge()
    self.plate_queue = []

    # Finds
    self.plate_reader = PlateReader(license_plate_model, id_plate_model, sess, graph)

    # Imitation models for the Outer loop, Intersections, and the Inner loop
    self.O = Imitator(outerloop_model, sess, graph)
    self.X = Imitator(intersection_model, sess, graph)
    self.I = Imitator(innerloop_model, sess, graph)

    self.state = 3
    self.count = 0
  

    print("Initialization complete")

    # Publish to plate0 to start the timer
    self.plates.publish('team_name,dogdoggo,0,D0OG')
    print("Published Start Message")
    
  
  def callback(self, data):
    sim_time = rospy.get_time() - self.init_time
  
    try:
      image = self.bridge.imgmsg_to_cv2(data, "passthrough")[-400:-1,:] # BGR
    except CvBridgeError as e:
      print(e)

    move = 0
    id_guess = 0
    plate = ["NO_PLATE"]

    # Outerloop 
    if self.state == 1:
      move = self.O.imitate(image)
      plate = self.plate_reader.identify(image)

      if self.pants(image):
        self.move_delay = 8

      self.move_delay = max(self.move_delay-1, 0)

      if self.move_delay > 0:
        move = 0

      # once we see P1 we can aim to drive into the innerloop
      if id_guess == 1:
        self.state = 2
        print("State: {}".format(self.state))


    # Intersection
    if self.state == 2:
      move = self.X.imitate(image)
      self.theta += self.move.angular.z

      # Truck detection
      if self.truck(image):
        self.move_delay = 30
      self.move_delay = max(self.move_delay-1, 0)
      if self.move_delay > 0:
        move = 0

      # Once we finish turning, start inner loop model
      if self.theta > 55:
        self.state = 3
        print("State: {}".format(self.state))

    # Inner Loop
    if self.state == 3:
      move = self.I.imitate(image)
      plate = self.plate_reader.identify(image)

      # Truck detection
      if self.truck(image):
        self.move_delay = 30
      self.move_delay = max(self.move_delay-1, 0)
      if self.move_delay > 0:
        move = 0

      if 7 == 8:
        self.state = 4
        self.plates.publish('team_name,dogdoggo,-1,D0OG')
        print("Published End Message")

    display = self.choose_move(move, image)


    if plate != "NO_PLATE":
      self.plate_queue.append(plate)
      cv2.imshow("Plate", plate)

    cv2.imshow("Debug Mode", display)
    cv2.waitKey(3)
    
    try:
      self.pub.publish(self.move)

    except CvBridgeError as e:
      print(e)

    self.count += 1
    if self.count % 2 == 0 and self.plate_queue:
      self.plate_queue.pop()

    print(len(self.plate_queue))

  def choose_move(self, move, image):
    x0 = image.shape[1]/2
    y0 = image.shape[0]*9/10
    delta = 200
    start = (x0, y0)

    if (move == 0):
      self.move.linear.x = 0
      self.move.angular.z = 0
      end = (x0, y0)
    if (move == 1):
      self.move.linear.x = 0
      self.move.angular.z = self.z
      end = (x0 - delta, y0)
    if (move == 2):
      self.move.linear.x = 0
      self.move.angular.z = -self.z
      end = (x0 + delta , y0)
    if (move == 3):
      self.move.linear.x = self.x
      self.move.angular.z = 0
      end = (x0, y0 - delta)
    if (move == 4):
      self.move.linear.x = self.x
      self.move.angular.z = self.z
      end = (x0 - delta, y0 - delta)
    if (move == 5):
      self.move.linear.x = self.x
      self.move.angular.z = -self.z
      end = (x0 + delta, y0 - delta)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return cv2.arrowedLine(image, start, end, (255, 0, 0), 9) 

  def pants(self, image):
   
    image = copy.copy(image[-300:-1,400:-400])
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array([80,30,20]), np.array([200,200,150]))
    pants = cv2.bitwise_and(image,image, mask= mask)
    kernel = np.ones((9, 9), np.uint8)
    erosion = cv2.erode(pants, kernel)
    
    h, w, _ = image.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, np.array([[[0,h],[0,100],[120,0],[w-120,0],[w,100],[w,h]]]), (255))
    crop = cv2.bitwise_and(erosion,erosion,mask = mask)

    if np.count_nonzero(crop) >= self.erosion_thresh:

      return True

    return False

  def truck(self, image):
    image = image[-300:-1,400:-400]
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    l_truck = np.array([160,0,0])
    u_truck = np.array([180,255,255])
    mask = cv2.inRange(hsv, l_truck, u_truck)
    res = cv2.bitwise_and(image,image, mask= mask)

    if np.count_nonzero(res) >= self.truck_thresh:

      return True

    return False

def main():
  rospy.init_node('controller', anonymous=True)
  rospy.sleep(1)
  controller = Controller()

  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
