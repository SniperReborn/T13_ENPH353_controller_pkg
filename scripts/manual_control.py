#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist

# source ~/ros_ws/devel/setup.bash
# cd ~/ros_ws/src/2025_competition/enph353/enph353_utils/scripts
# ./run_sim.sh -vpgw
# rosrun joy joy_node _dev:=/dev/input/js2 _autorepeat_rate:=20 _deadzone:=0.0

class ManualController:
    def __init__(self):
        rospy.init_node('manual_control_node')

        self.LINEAR_AXIS = 1   # Left Stick Up/Down
        self.ANGULAR_AXIS = 3 # Right Stick Left/Right
        
        # Max speeds
        self.linear_scale = 0.5  # m/s
        self.angular_scale = 2.25 # rad/s
        
        # Deadzone (Ignore inputs smaller than this value)
        # Increase this if the robot still drifts, decrease if it feels unresponsive
        self.deadzone = 0.05

        self.pub = rospy.Publisher('/B1/cmd_vel', Twist, queue_size=10)
        self.sub = rospy.Subscriber('/joy', Joy, self.joy_callback)
        
        rospy.loginfo("Manual Control Node Initialized (with deadzone filter).")

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