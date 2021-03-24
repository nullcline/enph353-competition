#!/usr/bin/env python

import sys
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

#tf_config = some_custom_config
sess = tf.Session()
graph = tf.get_default_graph()

# IMPORTANT: models have to be loaded AFTER SETTING THE SESSION for keras! 
# Otherwise, their weights will be unavailable in the threads after the session there has been set
set_session(sess)
OL_model = load_model('/home/andrew/ros_ws/src/2020T1_competition/controller/models/OL2')

class controller:

  def __init__(self):
    
    # CNNs
    # self.OL_model = tf.keras.models.load_model('/home/andrew/ros_ws/src/2020T1_competition/controller/models/OL.h5')
    # self.graph = tf.get_default_graph()

    # Publish plate 0 to start scoring
    self.init_time = rospy.get_time()
    self.plates = rospy.Publisher('/license_plate', String, queue_size=1)
    rospy.sleep(1)
    

    # publisher for movement
    self.pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)
    self.move = Twist()
    self.x = 0.15
    self.z = 0.5
    
    # image stuff
    self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image, self.callback, queue_size=1, buff_size=2**24) 
    self.bridge = CvBridge()
    self.complete = False
    self.queue = []

    print("Initialization complete")
    self.plates.publish('Test_Team,dogdoggo,0,D0OG')
    print("Published Start Message")
    
  
  def callback(self, data):

    # tf.reset_default_graph()
    # thread_graph = tf.Graph()

    rate = rospy.Rate(2)
    sim_time = rospy.get_time() - self.init_time
    # converting ros image to opencv 
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "passthrough")
    except CvBridgeError as e:
      print(e)

    # Normalization for input into imitation model
    cam = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    crop = cv_image[-400:-1,:]
    bw = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    res = cv2.resize(bw, dsize=(320,180))

    # add Frame to queue
    self.queue.append(res)

    if (len(self.queue) > 4):
      self.queue.pop(0)
      self.imitate()
    
    cv2.imshow("Raw Feed", cam)
    cv2.waitKey(3)

    # If we reach max time or we get all plates, say we're done
    if(sim_time > 40.0 and self.complete == False):
      self.complete = True
      self.plates.publish('Test_Team,dogdoggo,-1,D0OG')
      print("Published End Message")
    
    try:
      self.pub.publish(self.move)
    except CvBridgeError as e:
      print(e)

  def imitate(self):

    # prep image for prediction
    img =  cv2.vconcat(self.queue)
    h, w = img.shape
    img_res = img.reshape(h, w, 1)
    img_aug = np.expand_dims(img_res, axis=0)

    move = 0

    # workaround for running model 
    global sess
    global graph
    with graph.as_default():
      set_session(sess)
      predict = OL_model.predict(img_aug)[0]
      move = np.argmax(predict)
    
    if (move == 0):
      self.move.linear.x = 0
      self.move.angular.z = 0
    if (move == 1):
      self.move.linear.x = 0
      self.move.angular.z = self.z
    if (move == 2):
      self.move.linear.x = 0
      self.move.angular.z = -self.z
    if (move == 3):
      self.move.linear.x = self.x
      self.move.angular.z = 0
    if (move == 4):
      self.move.linear.x = self.x
      self.move.angular.z = self.z
    if (move == 5):
      self.move.linear.x = self.x
      self.move.angular.z = -self.z


  def find_parking(self, img):
    # Takes section of image from camera and returns cropped image of just the plate and number
    #
    # Responsible for removing distortion, and undoing perspective warp to get an optimal picture for the CNN
    #
    # param: img - matrix representation of image from camera
    # returns: ret - Boolean for if a spot was dected within a threshold of area
    # returns: plate - Cropped and normalized image of license plate
    # returns: spot - Cropped and normalized image of parking spot

    # Catherine's code here

    ret = True
    plate = []
    spot = []
    return ret, plate, spot
  
  def pred_parking(self, plate, spot, plate_model, spot_model):
    # Predicts what a license plate says using a pretrained CNN
    #
    # param: crop - cropped and normalized image of license plate from find_plate()
    # param: model - CNN weight/architecture file or whatever
    # returns predicted license plate as a string

    # Use Catherine's models here

    return "ABCD", "1"

def main():
  rospy.init_node('controller', anonymous=True)
  rospy.sleep(1)
  ct = controller()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
