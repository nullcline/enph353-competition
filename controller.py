#!/usr/bin/env python

import rospy
import cv2
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String

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
outerloop_model     = load_model('/home/andrew/ros_ws/src/2020T1_competition/controller/models/OLv0.h5')
intersection_model  = load_model('/home/andrew/ros_ws/src/2020T1_competition/controller/models/OLv0.h5')
innerloop_model     = load_model('/home/andrew/ros_ws/src/2020T1_competition/controller/models/OLv0.h5')
license_plate_model = load_model('/home/andrew/ros_ws/src/2020T1_competition/controller/models/Pv1.h5')

class Controller:

  def __init__(self):

    # Publish plate 0 to start scoring
    self.init_time = rospy.get_time()
    self.plates = rospy.Publisher('/license_plate', String, queue_size=1)
    rospy.sleep(1)

    # publisher for movement
    self.pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)
    self.move = Twist()
    self.x = 0.5
    self.z = 2
    
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

    self.state = 0

    print("Initialization complete")

    self.plates.publish('team_name,dogdoggo,0,D0OG')
    print("Published Start Message")
    
  
  def callback(self, data):
    sim_time = rospy.get_time() - self.init_time
  
    try:
      image = self.bridge.imgmsg_to_cv2(data, "passthrough")[-400:-1,:] # BGR
    except CvBridgeError as e:
      print(e)

    move = 0
    
    if self.state == 0:
      move = self.X.imitate(image)
      guess = self.plate_reader.identify(image)

    if self.state == 1:
      move = self.O.imitate(image)
      guess = self.plate.identify(image)

    display = self.choose_move(move, image)

    if guess != None:
      #h, w, _ = cv2.hconcat(guess)
      #display = cv2.vconcat(display, guess)
      cv2.imshow("Dog", cv2.hconcat(guess))
      
    cv2.imshow("Debug Mode", display)
    cv2.waitKey(3)

    # If we reach max time or we get all plates, say we're done
    if(sim_time > 240.0 or self.complete == True):
      self.complete = True
      self.plates.publish('team_name,dogdoggo,-1,D0OG')
      print("Published End Message")
    
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

def main():
  rospy.init_node('controller', anonymous=True)
  rospy.sleep(1)
  ct = Controller()

  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
