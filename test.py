#!/usr/bin/env python

import sys
import rospy
import cv2
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError

class line_follower:

  def __init__(self):
    # publisher for movement, subscriber to get image
    self.pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,self.callback)

  def callback(self,data):
    # converting ros image to opencv 
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "passthrough")
    except CvBridgeError as e:
      print(e)

    # taking the bottom of the image 
    frame_bw = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    bottom = frame_bw[-200:-1,:]
    middle = bottom.shape[1]/2

    # turning image binary
    ret, thresh = cv2.threshold(bottom,200,255,cv2.THRESH_BINARY_INV)
    moments = cv2.moments(thresh)

    # finding COM
    x = middle
    if(moments["m00"] != 0):
      x = int(moments["m10"]/moments["m00"])

    # troubleshooting stuff
    cv2.putText(thresh, str(x-middle), (x+10, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 2)
    cv2.circle(thresh, (x, 50), 5, (100, 100, 100), -1)
    cv2.imshow("Image window", thresh)
  
    cv2.waitKey(3)

    rate = rospy.Rate(2)
    move = Twist()
    move.linear.x = 0.5

    #proportionally turn towards the middle. 4 is just a scaling factor i found worked well.
    move.angular.z = -4*(x-middle)/middle

    try:
      print("reeeee")
      self.pub.publish(move)
    except CvBridgeError as e:
      print(e)

def main():
  lf = line_follower()
  rospy.init_node('controller', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
