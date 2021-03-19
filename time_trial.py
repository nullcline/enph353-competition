#!/usr/bin/env python

import sys
import rospy
import cv2
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String

class controller:

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

  
  def callback(self, data):

    sim_time = rospy.get_time() - self.init_time
    # converting ros image to opencv 
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "passthrough")
      cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    except CvBridgeError as e:
      print(e)

    # cv2.imshow("Image window", cv_image)
    # cv2.waitKey(3)

    rate = rospy.Rate(2)
    move = Twist()

    if(sim_time > 20.0 and sim_time < 30.0): 
      move.angular.z = 0.5
    else:
      move.linear.x = 0.1

    if(sim_time > 40.0 and self.complete == False):
      self.complete = True
      print("hi")
      self.plates.publish('Test_Team,dogdoggo,-1,D0OG')
    
    try:
      print(sim_time)
      self.pub.publish(move)
    except CvBridgeError as e:
      print(e)

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
