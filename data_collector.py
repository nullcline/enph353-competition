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

    # last 4 images, front of queue is the oldest of the 4
    self.queue = []

    with Listener(on_press=self.on_press, on_release=self.on_release) as listener:
      listener.join()

  # Runs everytime subscriber reads an image
  def callback(self, data):
    
    sim_time = rospy.get_time() - self.init_time
    rate = rospy.Rate(10)

    # converting ros image to opencv 
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "passthrough")
    except CvBridgeError as e:
      print(e)


    cam = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    crop = cv_image[-400:-1,:]
    bw = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    res = cv2.resize(bw, dsize=(320,180))

    
    # add frame to queue
    self.queue.append(res)

    if (len(self.queue) > 4):
      self.queue.pop(0)

    img_data = cv2.vconcat(self.queue)

    # cv2.imshow("Raw Feed", cam)
    cv2.imshow("End of Queue", img_data)
    cv2.waitKey(3)

    if(sim_time > 60.0*4 and self.complete == False):
      self.complete = True
      self.plates.publish('Test_Team,dogdoggo,-1,D0OG')
      print("Published Stop Code !")
    
    try:
      self.pub.publish(self.move)
    except CvBridgeError as e:
      print(e)

  # two methods below are responsible for reading keyboard and setting velocity values
  def on_press(self, key):

    try:
      #print("pressed {}".format(key.char))

      if (key.char == 'w'):
        self.move.linear.x = 0.15
      if (key.char == 'a'):
        self.move.angular.z = 0.5
      if (key.char == 'd'):
        self.move.angular.z = -0.5

    except: 
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
