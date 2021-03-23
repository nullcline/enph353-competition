#!/usr/bin/env python

import sys
import rospy
import cv2
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String

from pynput.keyboard import Key, Listener

class data_collector:

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

    with Listener(on_press=self.on_press, on_release=self.on_release) as listener:
      listener.join()

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


    cv2.imshow("Cam", cv_image)
    cv2.waitKey(3)

    if(sim_time > 60.0*4 and self.complete == False):
      self.complete = True
      self.plates.publish('Test_Team,dogdoggo,-1,D0OG')
      print("Published Stop Code !")
    
    try:
      self.pub.publish(self.move)
    except CvBridgeError as e:
      print(e)

  def on_press(self, key):
    try:
      print("pressed {}".format(key.char))

      if (key.char == 'w'):
        self.move.linear.x = 0.1

    except: 
      pass

  def on_release(self, key):
    try:
      print("released {}".format(key.char))

      if (key.char == 'w'):
        self.move.linear.x = 0

    except:
      pass

    if key == Key.esc:
        # Stop listener
        return False

def main():
  rospy.init_node('data_collector', anonymous=True)
  rospy.sleep(1)

  dc = data_collector()

  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
