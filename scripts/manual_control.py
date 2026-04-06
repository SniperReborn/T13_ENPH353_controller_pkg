#!/usr/bin/env python3

import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Joy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

# source ~/ros_ws/devel/setup.bash
# cd ~/ros_ws/src/2025_competition/enph353/enph353_utils/scripts
# ./run_sim.sh -vpgw
# rosrun joy joy_node _dev:=/dev/input/js2 _autorepeat_rate:=20 _deadzone:=0.0

class ManualController:
    def __init__(self):
        rospy.init_node('manual_control_node')
        self.bridge = CvBridge()

        self.LINEAR_AXIS = 1   # Left Stick Up/Down
        self.ANGULAR_AXIS = 3  # Right Stick Left/Right
        
        # Max speeds
        self.linear_scale = 0.5   # m/s
        self.angular_scale = 2.25 # rad/s
        
        # Deadzone
        self.deadzone = 0.05

        self.pub = rospy.Publisher('/B1/cmd_vel', Twist, queue_size=10)
        
        # FIX 1: Split into two separate callbacks
        self.sub = rospy.Subscriber('/joy', Joy, self.joy_callback)
        self.sub_camera = rospy.Subscriber('/B1/rrbot/camera1/image_raw', Image, self.image_callback)
        
        rospy.loginfo("Manual Control Node Initialized (with deadzone filter).")

    def image_callback(self, data):
        # Extract camera feed
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return
            
        # Extract image properties
        h, w, _ = cv_image.shape

        # FIX 3: Keep in BGR for display, or convert to actual HSV if you plan 
        # to do computer vision operations on it (cv2.COLOR_BGR2HSV). 
        # For now, we will just crop the standard BGR image so it displays correctly.
        cropped_image = cv_image[int(0.33*h):h, :]

        cv2.imshow("Camera View", cropped_image)
        
        # FIX 2: waitKey is required to refresh the OpenCV GUI window
        cv2.waitKey(1)

    def joy_callback(self, data):
        twist = Twist()
        
        # 1. Read the raw analog stick values
        raw_linear = data.axes[self.LINEAR_AXIS]
        raw_angular = data.axes[self.ANGULAR_AXIS]
        
        # 2. Apply the deadzone filter
        if abs(raw_linear) < self.deadzone:
            raw_linear = 0.0
            
        if abs(raw_angular) < self.deadzone:
            raw_angular = 0.0
            
        # 3. Scale the filtered values and assign them to the message
        twist.linear.x = raw_linear * self.linear_scale
        twist.angular.z = raw_angular * self.angular_scale

        self.pub.publish(twist)

if __name__ == '__main__':
    try:
        ManualController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass