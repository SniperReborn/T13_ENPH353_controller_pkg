#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

# Import your custom modules
from eyes import PinkDetector
from brains import TFLiteBrain

class RobotState:
    MODEL_1 = 0
    PAUSED = 1
    MODEL_2 = 2

class StateMachineController:
    def __init__(self):
        rospy.init_node('state_machine_driver')
        self.bridge = CvBridge()
        
        # 1. Initialize Subsystems
        self.eyes = PinkDetector()
        
        # Load BOTH models right at the start
        self.brain_1 = TFLiteBrain('~/cnn_trainer/pavement_model.tflite')
        
        # TODO: Change this path to your second model when it's ready!
        self.brain_2 = TFLiteBrain('~/cnn_trainer/dirt_model.tflite') 
        
        # 2. Setup ROS
        self.image_sub = rospy.Subscriber("/B1/rrbot/camera1/image_raw", Image, self.image_callback)
        self.cmd_pub = rospy.Publisher("/B1/cmd_vel", Twist, queue_size=1)
        
        # 3. State Machine Variables
        self.current_state = RobotState.MODEL_1
        self.pause_start_time = None

        # 4. Initialize to a long time ago so we don't trigger a cooldown on startup
        self.cooldown_finish_time = rospy.Time(0) 
        self.cooldown_duration = 2.0
        
        rospy.loginfo("State Machine Initialized. Running Model 1.")

    def publish_velocity(self, lin_x, ang_z):
        move_cmd = Twist()
        move_cmd.linear.x = lin_x
        move_cmd.angular.z = ang_z
        self.cmd_pub.publish(move_cmd)

    def image_callback(self, data):
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")      

        current_time = rospy.Time.now()
        is_in_cooldown = (current_time - self.cooldown_finish_time).to_sec() < self.cooldown_duration
        
        # STATE 0: Driving with Pavement Model
        if self.current_state == RobotState.MODEL_1:
            if not is_in_cooldown and self.eyes.is_pink_line_present(cv_image):
                rospy.loginfo("PINK DETECTED! Stopping.")
                self.current_state = RobotState.PAUSED
                self.pause_start_time = current_time
                self.publish_velocity(0.0, 0.0) # Hit the brakes
            else:
                lin_x, ang_z = self.brain_1.get_command(cv_image)
                self.publish_velocity(lin_x, ang_z)

        # STATE 1: Paused (Waiting for 1 second)
        elif self.current_state == RobotState.PAUSED:
            elapsed = (current_time - self.pause_start_time).to_sec()
            self.publish_velocity(0.0, 0.0) # Ensure we stay stopped
            
            if elapsed >= 1.0:
                rospy.loginfo("Pause complete. Switching to Model 2.")
                self.current_state = RobotState.MODEL_2
                self.cooldown_finish_time = current_time
                
        # STATE 2: Driving with Model 2
        elif self.current_state == RobotState.MODEL_2:
            if not is_in_cooldown and self.eyes.is_pink_line_present(cv_image):
                rospy.loginfo("PINK DETECTED! Stopping.")
                self.current_state = RobotState.PAUSED
                self.pause_start_time = rospy.Time.now()
                self.publish_velocity(0.0, 0.0) # Hit the brakes
            else:
                lin_x, ang_z = self.brain_2.get_command(cv_image)
                self.publish_velocity(lin_x, ang_z)

if __name__ == '__main__':
    try:
        driver = StateMachineController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass