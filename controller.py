#!/usr/bin/env python
#encoding: utf-8

import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings("ignore")

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import rospy
import cv2
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
from std_msgs.msg import Time
import copy
import time
import numpy as np

from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model

from plate_reader import PlateReader
from imitator import Imitator

# Required setup for running the models in image callback
sess = tf.Session()
graph = tf.get_default_graph()
set_session(sess)
outerloop_model     = load_model('./models/OLv1.h5')
intersection_model  = load_model('./models/Xv2.h5')
innerloop_model     = load_model('./models/ILv0.h5')
license_plate_model = load_model('./models/plate_number_model_v4.h5')
id_plate_model      = load_model('./models/plate_id_model_v3.h5')

class Controller:

  def __init__(self):

  
    self.init_time = rospy.get_time()
    self.plates = rospy.Publisher('/license_plate', String, queue_size=1)
    rospy.sleep(1)

    # publisher for movement
    self.pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)
    self.move = Twist()
    self.move_delay = 0
    self.x = 0.25
    self.z = 1
    self.theta = 0
    self.erosion_thresh = 100
    self.truck_thresh = 50
    self.blur_thresh = 190
    self.slow = False
    
    # image stuff
    self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image, self.callback, queue_size=1, buff_size=2**24) 
    self.bridge = CvBridge()

    # Finds
    self.plate_reader = PlateReader(license_plate_model, id_plate_model, sess, graph)
    self.plate_delay = 0
    self.id = 0
    self.id_order = [2, 3, 4, 5, 6, 1, 7, 8, -1]
    self.guess = {0 : (["N","U","L","L"],[0,0,0,0]),
                  1 : (["N","U","L","L"],[0,0,0,0]),
                  2 : (["N","U","L","L"],[0,0,0,0]),
                  3 : (["N","U","L","L"],[0,0,0,0]),
                  4 : (["N","U","L","L"],[0,0,0,0]),
                  5 : (["N","U","L","L"],[0,0,0,0]),
                  6 : (["N","U","L","L"],[0,0,0,0]),
                  7 : (["N","U","L","L"],[0,0,0,0])}

    # Imitation models for the Outer loop, Intersections, and the Inner loop
    self.O = Imitator(outerloop_model, sess, graph)
    self.X = Imitator(intersection_model, sess, graph)
    self.I = Imitator(innerloop_model, sess, graph)

    self.state = 0
  
    # Run the models once cause it lags a lot the first time?
    tmp = cv2.imread("dog.png")
    tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
    _, tmp, tmp1 = self.plate_reader.find(tmp)
    tmp = self.plate_reader.guess(tmp, tmp1)

    print("Initialization complete")

    # Publish to plate0 to start the timer
    self.plates.publish('2 Shades of Grey, hunter2 ,0,OZZY')
    print("Published Start Message")
    
  
  def callback(self, data):
    sim_time = rospy.get_time() - self.init_time
  
    try:
      image = self.bridge.imgmsg_to_cv2(data, "passthrough")[-400:-1,:] # BGR
    except CvBridgeError as e:
      print(e)

    move = 0
    plate = "NO_PLATE"

    # Starting 
    if self.state == 0:
      move = self.X.imitate(image)
      plate, plate_chars, plate_id = self.plate_reader.find(image)

      if plate != "NO_PLATE":
        print("State: {}".format(self.state))
        self.state = 1

    # Outerloop 
    if self.state == 1:
      move = self.O.imitate(image)
      plate, plate_chars, plate_id = self.plate_reader.find(image)

      # "Graceful" stopping for pedestrians
      if self.pants(image):
        self.move_delay = 10.0

      self.move_delay = max(self.move_delay-1, 0)

      if self.move_delay > 0:
        move = 0

      # Estimating when to turn into inner loop
      self.theta += self.move.angular.z

      # We have a hard time detecting 1 cause of low lapacian variance or whatever
      if self.id == 5:
        self.blur_thresh = 130

      if self.id == 6 or self.theta > 120:
        self.blur_thresh = 190
        self.state = 2
        self.theta = 0
        self.x = 0.5
        self.z = 2
        print("State: {}".format(self.state))


    # Intersection
    if self.state == 2:
      move = self.X.imitate(image)

      # Truck detection -> If we're behind the truck we might as well slowdown forever :(
      if self.truck(image):
        self.x = 0.12
        self.z = 0.4

      # Once we finish turning, start inner loop model
      self.theta += self.move.angular.z

      if self.theta > 50:
        self.state = 3
        self.theta = 0
        print("State: {}".format(self.state))

    # Inner Loop
    if self.state == 3:
      move = self.I.imitate(image)
      plate, plate_chars, plate_id = self.plate_reader.find(image)

      # Truck detection -> If we're behind the truck we might as well slowdown forever :(
      if self.truck(image):
        self.x = 0.12
        self.z = 0.5

      # Once we've turned enough to read plate 8, we're done.
      self.theta += self.move.angular.z

      if self.id == 8:
        
        self.state = 4
        self.plates.publish('2 Shades of Grey, hunter2 ,-1,OZZY')
        print("Published End Message")
        cv2.destroyAllWindows()
    
    if self.state == 4:
      # Crucial, do not remove
      os.system('clear' if os.name == 'posix' else 'CLS')

      while(True):

        print("░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░")
        print("░░░░░░░░░░░▄▀▄▀▀▀▀▄▀▄░░░░░░░░░░░░░░░░░░")
        print("░░░░░░░░░░░█░░░░░░░░▀▄░░░░░░▄░░░░░░░░░░")
        print("░░░░░░░░░░█░░▀░░▀░░░░░▀▄▄░░█░█░░░░░░░░░")
        print("░░░░░░░░░░█░▄░█▀░▄░░░░░░░▀▀░░█░░░░░░░░░")
        print("░░░░░░░░░░█░░▀▀▀▀░░░░░░░░░░░░█░░░░░░░░░")
        print("░░░░░░░░░░█░░░░░░░░░░░░░░░░░░█░░░░░░░░░")
        print("░░░░░░░░░░█░░░░░░░░░░░░░░░░░░█░░░░░░░░░")
        print("░░░░░░░░░░░█░░▄▄░░▄▄▄▄░░▄▄░░█░░░░░░░░░░")
        print("░░░░░░░░░░░█░▄▀█░▄▀░░█░▄▀█░▄▀░░░░░░░░░░")
        print("░complete░░░▀░░░▀░░░░░▀░░░▀░░░░░░░░░░░░")
        print("░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░")

        time.sleep(1)
        os.system('clear' if os.name == 'posix' else 'CLS')

        print("░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░")
        print("░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░")
        print("░░░░░░░░░░░▄▀▄▀▀▀▀▄▀▄░░░░░░░░░░░░░░░░░░")
        print("░░░░░░░░░░░█░░░░░░░░▀▄░░░░░░▄░░░░░░░░░░")
        print("░░░░░░░░░░█░░▀░░▀░░░░░▀▄▄░░█░█░░░░░░░░░")
        print("░░░░░░░░░░█░▄░█▀░▄░░░░░░░▀▀░░█░░░░░░░░░")
        print("░░░░░░░░░░█░░▀▀▀▀░░░░░░░░░░░░█░░░░░░░░░")
        print("░░░░░░░░░░█░░░░░░░░░░░░░░░░░░█░░░░░░░░░")
        print("░░░░░░░░░░█░░░░░░░░░░░░░░░░░░█░░░░░░░░░")
        print("░░░░░░░░░░░█░▄▀█░▄▀▀▀█░▄▀█░▄▀░░░░░░░░░░")
        print("░complete░░░▀░░░▀░░░░░▀░░░▀░░░░░░░░░░░░")
        print("░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░")

        time.sleep(1)
        os.system('clear' if os.name == 'posix' else 'CLS')

    # Changes self.move values to match imitation, also edits display to show a cool arrow
    display = self.choose_move(move, image)
    if plate != "NO_PLATE" and cv2.Laplacian(plate, cv2.CV_64F).var() > self.blur_thresh:
      
      self.plate_delay = 30
      string_guess, string_prob, id_guess, id_prob = self.plate_reader.guess(plate_chars, plate_id)
      #print(string_prob)
      # guess[plate_id] gives a pair, (["N","U","L","L"], [P1,P2,P3,P4])

      for i, char in enumerate(string_guess):

        if (string_prob[i] > self.guess[self.id][1][i]):
          self.guess[self.id][0][i] = char
          self.guess[self.id][1][i] = string_prob[i]

      # cv2.imshow("Plate", plate)
      # cv2.imshow("Chars", cv2.hconcat(plate_chars))
      
    self.plate_delay = max(self.plate_delay-1, 0)
    if self.plate_delay == 1:
      print("Guess for ID:{} ---- {}".format(self.id_order[self.id], self.guess[self.id][0]))
      self.plates.publish('2 Shades of Grey, hunter2 ,{},{}'.format(self.id_order[self.id],"".join(self.guess[self.id][0])))
      self.id += 1
    
    # cv2.imshow("Debug Mode", display)
    # cv2.waitKey(3)
    
    try:
      self.pub.publish(self.move)

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

    image = copy.copy(image[-300:-1,400:-400])
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
