#!/usr/bin/env python

import rospy
import cv2
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
from std_msgs.msg import Time

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
license_plate_model = load_model('/home/andrew/ros_ws/src/2020T1_competition/controller/models/Pv0.h5')

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
    
    # image stuff
    self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image, self.callback, queue_size=1, buff_size=2**24) 
    self.bridge = CvBridge()
    self.prediction = ""
    self.complete = False

    # Finds
    self.plate_reader = PlateReader(license_plate_model, sess, graph)

    # Imitation models for the Outer loop, Intersections, and the Inner loop
    self.O = Imitator(outerloop_model, sess, graph)
    self.X = Imitator(intersection_model, sess, graph)
    self.I = Imitator(innerloop_model, sess, graph)

    self.state = 1
  

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
    plate = ["NO_PLATE"]

    # Leave starting position
    if self.state == 0:
      move = self.X.imitate(image)
      plate, guess, probs = self.plate_reader.identify(image)

      # Ideally this activates of reading the first plate
      if plate[0] != "NO_PLATE":
        self.state = 1
        self.tehta = 0

    # Outerloop 
    if self.state == 1:
      move = self.O.imitate(image)
      plate, guess, probs = self.plate_reader.identify(image)

      if self.pants(image):
        print("here")
        self.move_delay = 5

      self.move_delay = max(self.move_delay-1, 0)
      print(self.move_delay)

      if self.move_delay > 0:
        move = 0
      # Ideally this activates off reading the 6th outer plate (P1)
      # if self.theta >= 125:
      #   self.state = 2
      #   self.theta = 0


    # Once we reach P1, turn into inner loop
    if self.state == 2:
      move = self.X.imitate(image)

      if self.theta > 55:
        self.state = 3

    # Once we finish turning, navigate the inner loop
    if self.state == 3:
      move = self.I.imitate(image)
      plate, guess, probs = self.plate_reader.identify(image)

    display = self.choose_move(move, image)


    if plate[0] != "NO_PLATE":
      print("Guess: {}", guess)
      cv2.imshow("Plate", cv2.hconcat(plate))

    cv2.imshow("Debug Mode", display)
    cv2.waitKey(3)

    # If we reach max time or we get all plates, say we're done
    if(sim_time > 240.0 or self.complete == True):
      self.complete = True
      self.plates.publish('team_name,dogdoggo,-1,D0OG')
      print("Published End Message")
    
    try:
      self.pub.publish(self.move)
      self.theta += self.move.angular.z
      # print("State: {}".format(self.state))
      # print(self.theta)

    except CvBridgeError as e:
      print(e)

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
   
    image = image[-300:-1,400:-400]
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

    pass

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
