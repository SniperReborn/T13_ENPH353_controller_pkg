#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

# Driving Imports
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
        
        model_paths = [
            '~/cnn_trainer/data_RTF=0.95/pavement_modelv2.tflite',
            '~/cnn_trainer/data_RTF=0.95/dirt_model.tflite',
            '~/cnn_trainer/data_RTF=0.95/yoda_modelv4.tflite',
            '~/cnn_trainer/data_RTF=0.95/mountain_modelv3.tflite'
        ]
        
        self.brains = [TFLiteBrain(path) for path in model_paths]
        self.current_brain_index = 0
        
        self.current_state = RobotState.DRIVING
        self.pause_start_time = None
        self.is_movement_detected = False
        self.cooldown_finish_time = rospy.Time(0) 
        self.cooldown_duration = 2.0

        self.angular_nudge = 1.0
        self.velocity_nudge = 0.9

        self.global_start_time = rospy.get_time()
        self.max_timeout_duration = 240.0 
        self.timeout_triggered = False

        self.cmd_pub = rospy.Publisher("/B1/cmd_vel", Twist, queue_size=1)
        self.score_publisher = rospy.Publisher('/score_tracker', String, queue_size=1)

        while self.score_publisher.get_num_connections() == 0:
            rospy.loginfo("Waiting for scoreboard subscriber...")
            rospy.sleep(0.1)

        self.score_publisher.publish("Team13,multi21,0,NA")
        
        # Subscribes to the new topic from the Sign Reader Node
        self.sign_sub = rospy.Subscriber("/B1/detected_signs", String, self.sign_callback, queue_size=10)
        
        # Main driving camera (RAW, fast, massive buffer)
        self.image_sub = rospy.Subscriber("/B1/rrbot/camera1/image_raw", Image, self.image_callback, queue_size=1, buff_size=2**24)
        self.lidar_sub = rospy.Subscriber("/scan", LaserScan, self.lidar_callback) 
        
        rospy.loginfo("Driver Node Initialized.")

    def sign_callback(self, msg):
        """Receives signs from the reader node and handles scoreboard/stopping logic."""
        sign_id_str, clue_string = msg.data.split(',')
        sign_id = int(sign_id_str)

        score_msg = f"Team13,multi21,{sign_id},{clue_string}"
        self.score_publisher.publish(score_msg) 

        if sign_id == 8:
            rospy.logwarn("Final sign read. Stopping!")
            self.score_publisher.publish("Team13,multi21,-1,NA") 
            rospy.signal_shutdown("Track Complete.")

    def publish_velocity(self, lin_x, ang_z):
        move_cmd = Twist()
        move_cmd.linear.x = lin_x
        move_cmd.angular.z = ang_z
        self.cmd_pub.publish(move_cmd)

    def lidar_callback(self, data):
        if self.current_state == RobotState.PAUSED and self.ears.detect_movement(data):
            self.is_movement_detected = True

    def image_callback(self, data):
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")      
        current_time = rospy.Time.now()
        
        # Timeout Protocol Check
        if (rospy.get_time() - self.global_start_time) > self.max_timeout_duration and not self.timeout_triggered:
            self.score_publisher.publish("Team13,multi21,-1,NA") 
            self.timeout_triggered = True
            rospy.logwarn("Global timeout reached!")

        if self.current_state == RobotState.DRIVING:
            is_in_cooldown = (current_time - self.cooldown_finish_time).to_sec() < self.cooldown_duration
            
            if not is_in_cooldown and self.eyes.is_pink_line_present(cv_image):
                self.current_state = RobotState.PAUSED
                self.pause_start_time = current_time
                self.ears.reset_baseline()
                self.is_movement_detected = False
                self.publish_velocity(0.0, 0.0)
            else:
                lin_x, ang_z = self.brains[self.current_brain_index].get_command(cv_image)
                self.publish_velocity(lin_x * self.velocity_nudge, ang_z * self.angular_nudge)

        elif self.current_state == RobotState.PAUSED:
            self.publish_velocity(0.0, 0.0)
            if self.is_movement_detected:
                self.pause_start_time = current_time
                self.ears.reset_baseline()
                self.is_movement_detected = False
                
            if (current_time - self.pause_start_time).to_sec() >= 2.0:
                self.current_brain_index = (self.current_brain_index + 1) % len(self.brains)
                self.current_state = RobotState.DRIVING
                self.cooldown_finish_time = current_time

if __name__ == '__main__':
    try:
        controller = StateMachineController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass