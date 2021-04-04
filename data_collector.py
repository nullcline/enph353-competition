#!/usr/bin/env python

import sys
import rospy
import cv2
import time
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
from pynput.keyboard import Key, Listener

import tensorflow as tf

from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model

from plate_reader import PlateReader
sess = tf.Session()
graph = tf.get_default_graph()
set_session(sess)
license_plate_model     = load_model('/home/andrew/ros_ws/src/2020T1_competition/controller/models/Pv1.h5')

class DataCollector:

  def __init__(self):

    # Publish plate 0 to start scoring
    self.init_time = rospy.get_time()
    self.plates = rospy.Publisher('/license_plate', String, queue_size=1)
    rospy.sleep(1)
    self.plates.publish('Test_Team,dogdoggo,0,D0OG')
    print("Published Start Code !")

    # publisher for movement
    self.pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)
    self.move = Twist()
    
    # image stuff
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,self.callback)
    self.complete = False 

    # last 4 images, front of queue is the oldest of the 4
    self.queue = []
    self.write = False

    self.plate_reader = PlateReader(license_plate_model, sess, graph)

    with Listener(on_press=self.on_press, on_release=self.on_release) as listener:
      listener.join()

  # Runs everytime subscriber reads an image
  def callback(self, data):
    
    sim_time = rospy.get_time() - self.init_time
    rate = rospy.Rate(2)

    # converting ros image to opencv 
    try:
      image = self.bridge.imgmsg_to_cv2(data, "passthrough")[-400:-1,:]
    except CvBridgeError as e:
      print(e)

    cam = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ret, thresh = cv2.threshold(image,200,255,cv2.THRESH_BINARY)
    bw = cv2.cvtColor(thresh, cv2.COLOR_RGB2GRAY)
    input_data = cv2.resize(bw, dsize=(160,90))

    plates, guess, probs  = self.plate_reader.identify(image)

    if plates[0] != "NO_PLATE":
      cv2.imshow("Plate", cv2.hconcat(plates))
      print("Guessed: {} with probablilities {}".format(guess, probs))

    if (self.write):
      cv2.putText(cam,'R',(20,90), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
      
    cv2.imshow("Raw Feed", cam)
    cv2.imshow("Input Data", input_data)
    cv2.waitKey(3)

    label = ""
    if (self.move.linear.x > 0.0):
      label += "1"
    else:
      label += "0"
    
    if (self.move.angular.z < 0.0):
      label += "2"
    elif (self.move.angular.z > 0.0):
      label += "1"
    else:
      label += "0"

    if (sim_time > 5.0 and self.write):
      t = time.time()
      cv2.imwrite('/home/andrew/ros_ws/src/2020T1_competition/controller/prev/{}_{}.jpg'.format(label, t), input_data)
      print("Saved {}_{}.jpg'".format(label, t))

    try:
      self.pub.publish(self.move)
    except CvBridgeError as e:
      print(e)

  # two methods below are responsible for reading keyboard and setting velocity values
  def on_press(self, key):

    try:
      #print("pressed {}".format(key.char))

      if (key.char == 'w'):
        self.move.linear.x = 0.50
      if (key.char == 'a'):
        self.move.angular.z = 2
      if (key.char == 'd'):
        self.move.angular.z = -2
      if (key.char =='j'):
        self.write = False
      if (key.char == 'l'):
        self.write = True

    except Exception: 
      pass

  def on_release(self, key):

    try:
      #print("released {}".format(key.char))

      if (key.char == 'w'):
        self.move.linear.x = 0
      if (key.char == 'a'):
        self.move.angular.z = 0
      if (key.char == 'd'):
        self.move.angular.z = 0

    except Exception:
      pass

    if key == Key.esc:
        # Stop listener
        return False

def main():
  rospy.init_node('data_collector', anonymous=True)
  rospy.sleep(1)

  dc = DataCollector()

  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
