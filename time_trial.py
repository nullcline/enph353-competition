#!/usr/bin/env python

import sys
import rospy
import cv2
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String

class time_trial:

  def __init__(self):

    # Publish plate 0 to start scoring
    self.init_time = rospy.get_time()
    print("started at " + str(self.init_time))
    self.plates = rospy.Publisher('/license_plate', String, queue_size=1)
    rospy.sleep(1)
    self.plates.publish('Test_Team,dogdoggo,0,D0OG')

    # publisher for movement
    self.pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)
    
    # image stuff
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,self.callback)
    self.complete = False

  # Runs everytime subscriber reads an image
  def callback(self, data):
    
    sim_time = rospy.get_time() - self.init_time

    # converting ros image to opencv 
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "passthrough")
      cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    except CvBridgeError as e:
      print(e)

    rate = rospy.Rate(10)
    move = Twist()

    
    ret, plate, spot = find_parking(cv_image)

    if (ret):
      plate_pr, spot_pr = pred_parking(plate, spot, None, None) 

    # maybe store the plate predictions and spot predictions in an array until find_parking stops, and publish the most common
    if (False):
      self.plates.publish(team, pswd, spot_pr, plate_pr)

    # hardcoded sequence to test image rec
    if(sim_time > 0 and sim_time <= 6.5): 
      move.linear.x = 0.1
    if(sim_time > 6.5 and sim_time <= 7.0): 
      move.angular.z = 3.5
    if(sim_time > 8.0 and sim_time <= 18.0): 
      move.linear.x = 0.1
    if(sim_time > 30.0 and sim_time <= 40.0): 
      pass
    if(sim_time > 40.0 and sim_time <= 50.0): 
      pass
   

    if(sim_time > 40.0 and self.complete == False):
      self.complete = True
      print("hi")
      self.plates.publish('Test_Team,dogdoggo,-1,D0OG')
    
    try:
      print(sim_time)
      self.pub.publish(move)
    except CvBridgeError as e:
      print(e)

def find_parking(img):
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
  
def pred_parking(plate, spot, plate_model, spot_model):
  # Predicts what a license plate says using a pretrained CNN
  #
  # param: crop - cropped and normalized image of license plate from find_plate()
  # param: model - CNN weight/architecture file or whatever
  # returns predicted license plate as a string

  # Use Catherine's models here

  return "ABCD", "1"

def main():
  rospy.init_node('time_trial', anonymous=True)
  rospy.sleep(1)
  tt = time_trial()

  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
