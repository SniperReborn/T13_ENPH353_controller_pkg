#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2

from eyes import PinkDetector
from brains import TFLiteBrain
from scanner import LidarMotionDetector

class RobotState:
    DRIVING = 0
    PAUSED = 1

class StateMachineController:
    def __init__(self):
        rospy.init_node('state_machine_driver')
        self.bridge = CvBridge()
        
        self.eyes = PinkDetector()
        self.ears = LidarMotionDetector(check_distance=10.0) 
        self.latest_left_image = None
        self.latest_right_image = None
        
        model_paths = [
            '~/cnn_trainer/pavement_modelv2.tflite',
            '~/cnn_trainer/dirt_modelv3.tflite',
            '~/cnn_trainer/yoda_modelv4.tflite',
            '~/cnn_trainer/mountain_modelv3.tflite'
        ]
        
        self.brains = []
        for path in model_paths:
            self.brains.append(TFLiteBrain(path))
            
        self.current_brain_index = 0
        
        self.image_sub = rospy.Subscriber("/B1/rrbot/camera1/image_raw", Image, self.image_callback, queue_size=1, buff_size=2**24)
        self.left_camera_sub = rospy.Subscriber('/B1/rrbot/camera_left/image_raw', Image, self.left_camera_callback, queue_size=1, buff_size=2**24)
        self.right_camera_sub = rospy.Subscriber('/B1/rrbot/camera_right/image_raw', Image, self.right_camera_callback, queue_size=1, buff_size=2**24)
    
        self.lidar_sub = rospy.Subscriber("/scan", LaserScan, self.lidar_callback)
        self.cmd_pub = rospy.Publisher("/B1/cmd_vel", Twist, queue_size=1)
        
        self.current_state = RobotState.DRIVING
        self.pause_start_time = None
        self.is_movement_detected = False
        self.cooldown_finish_time = rospy.Time(0) 
        self.cooldown_duration = 2.0
        
        rospy.loginfo(f"State Machine Initialized. Running Model {self.current_brain_index + 1}.")

    def publish_velocity(self, lin_x, ang_z):
        move_cmd = Twist()
        move_cmd.linear.x = lin_x
        move_cmd.angular.z = ang_z
        self.cmd_pub.publish(move_cmd)

    def lidar_callback(self, data):
        if self.current_state == RobotState.PAUSED:
            if self.ears.detect_movement(data):
                self.is_movement_detected = True

    def left_camera_callback(self, data):
        self.latest_left_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

    def right_camera_callback(self, data):
        self.latest_right_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

    def image_callback(self, data):
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")      
        current_time = rospy.Time.now()

        # --- STATE: DRIVING ---
        if self.current_state == RobotState.DRIVING:
            is_in_cooldown = (current_time - self.cooldown_finish_time).to_sec() < self.cooldown_duration
            
            if not is_in_cooldown and self.eyes.is_pink_line_present(cv_image):
                rospy.loginfo("PINK DETECTED! Stopping.")
                self.current_state = RobotState.PAUSED
                self.pause_start_time = current_time
                self.ears.reset_baseline()
                self.is_movement_detected = False
                self.publish_velocity(0.0, 0.0)
            else:
                lin_x, ang_z = self.brains[self.current_brain_index].get_command(cv_image)
                self.publish_velocity(lin_x * 0.7, ang_z)

        # --- STATE: PAUSED ---
        elif self.current_state == RobotState.PAUSED:
            self.publish_velocity(0.0, 0.0)
            
            # If movement was flagged, reset the timer and baseline
            if self.is_movement_detected:
                rospy.loginfo("Movement detected! Resetting 1-second timer.")
                self.pause_start_time = current_time
                self.ears.reset_baseline()
                self.is_movement_detected = False
                
            elapsed = (current_time - self.pause_start_time).to_sec()
            
            # Once 1 second has cleanly passed with NO movement interruptions
            if elapsed >= 2.0:
                self.current_brain_index = (self.current_brain_index + 1) % len(self.brains)
                rospy.loginfo(f"All clear. Switching to Model {self.current_brain_index + 1}.")
                self.current_state = RobotState.DRIVING
                self.cooldown_finish_time = current_time

    def render_loop(self):
        rate = rospy.Rate(30) # Refresh the GUI at 30Hz
        while not rospy.is_shutdown():
            if self.latest_left_image is not None:
                cv2.imshow("Feed Left", self.latest_left_image)
            if self.latest_right_image is not None:
                cv2.imshow("Feed Right", self.latest_right_image)
            
            # Optional: Uncomment if you also want to see the front camera
            # if self.latest_front_image is not None:
            #     cv2.imshow("Feed Front", self.latest_front_image)
            
            cv2.waitKey(1)
            rate.sleep()

if __name__ == '__main__':
    try:
        controller = StateMachineController()
        controller.render_loop()
    except rospy.ROSInterruptException:
        pass