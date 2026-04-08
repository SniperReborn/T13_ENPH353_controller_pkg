#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2

# State Machine Imports
from eyes import PinkDetector
from brains import TFLiteBrain
from scanner import LidarMotionDetector

# Sign Reading Imports
from sign_extract import SignTracker
from letter_extract import extract_and_sort_letters
from ocr import SignReaderCNN

class RobotState:
    DRIVING = 0
    PAUSED = 1

class StateMachineController:
    def __init__(self):
        rospy.init_node('state_machine_driver')
        self.bridge = CvBridge()
        
        # --- Driving Components ---
        self.eyes = PinkDetector()
        self.ears = LidarMotionDetector(check_distance=10.0) 
        self.latest_left_image = None
        self.latest_right_image = None
        
        model_paths = [
            '~/cnn_trainer/data_RTF=0.95/pavement_model.tflite',
            '~/cnn_trainer/data_RTF=0.95/dirt_modelv3.tflite',
            '~/cnn_trainer/data_RTF=0.95/yoda_modelv4.tflite',
            '~/cnn_trainer/data_RTF=0.95/mountain_modelv3.tflite'
        ]

        self.topic_map = {
            "SIZE": 1,
            "VICTIM": 2,
            "CRIME": 3,
            "TIME": 4,
            "PLACE": 5,
            "MOTIVE": 6,
            "WEAPON": 7,
            "BANDIT": 8
        } 
        
        self.brains = []
        for path in model_paths:
            self.brains.append(TFLiteBrain(path))
            
        self.current_brain_index = 0
        
        # State tracking
        self.current_state = RobotState.DRIVING
        self.pause_start_time = None
        self.is_movement_detected = False
        self.cooldown_finish_time = rospy.Time(0) 
        self.cooldown_duration = 2.0
        self.angular_nudge = 1.0
        self.velocity_nudge = 0.9

        self.max_timeout_duration = 240.0 
        self.timeout_triggered = False
        
        # --- Sign Reading Components ---
        self.min_sign_size = 30000
        self.LOWER_B, self.UPPER_B = (115, 125, 100), (125, 255, 210)
        self.LOWER_W, self.UPPER_W = (0, 0, 100), (3, 3, 255)

        self.cnn = SignReaderCNN("/home/fizzer/cnn_trainer/wei_dynasty.tflite")
        
        self.active_sign = {'image': None, 'area': 0, 'timestamp': 0.0}
        self.left_tracker = SignTracker(self.LOWER_B, self.UPPER_B, self.LOWER_W, self.UPPER_W, self.min_sign_size, 3.0)
        self.right_tracker = SignTracker(self.LOWER_B, self.UPPER_B, self.LOWER_W, self.UPPER_W, self.min_sign_size, 3.0)

        self.comparison_window = 1.0  
        self.sign_processed = False   
        self.sign_window_open = False

        # --- Publishers & Subscribers ---
        self.cmd_pub = rospy.Publisher("/B1/cmd_vel", Twist, queue_size=1)
        self.score_publisher = rospy.Publisher('/score_tracker', String, queue_size=1)

        while self.score_publisher.get_num_connections() == 0:
            rospy.loginfo("Waiting for scoreboard subscriber...")
            rospy.sleep(0.1)

        self.global_start_time = rospy.get_time()

        first_msg = "Team13,multi21,0,NA"
        self.score_publisher.publish(first_msg)
        rospy.loginfo("Published first message") 
        
        self.image_sub = rospy.Subscriber("/B1/rrbot/camera1/image_raw", Image, self.image_callback, queue_size=1)
        self.left_camera_sub = rospy.Subscriber('/B1/rrbot/camera_left/image_raw', Image, self.left_camera_callback, queue_size=1)
        self.right_camera_sub = rospy.Subscriber('/B1/rrbot/camera_right/image_raw', Image, self.right_camera_callback, queue_size=1)
        self.lidar_sub = rospy.Subscriber("/scan", LaserScan, self.lidar_callback) 
        
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

    def sync_best_sign(self, img, area):
        now = rospy.get_time()
        time_since_last = now - self.active_sign['timestamp']
        
        # 1. NEW SIGN
        if time_since_last > 3.0:
            self.active_sign = {'image': img, 'area': area, 'timestamp': now}
            rospy.loginfo("First camera locked a sign. Comparison window OPEN.")

        # 2. COMPARISON
        elif time_since_last < self.comparison_window:
            if area > self.active_sign['area']:
                self.active_sign['image'] = img
                self.active_sign['area'] = area
                rospy.loginfo(f"Found a better view! New Area: {area}")

    def left_camera_callback(self, data):
        self.latest_left_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        res = self.left_tracker.update(self.latest_left_image, rospy.get_time())
        if res is not None: 
            self.sync_best_sign(res, self.left_tracker.max_sign_area)

    def right_camera_callback(self, data):
        self.latest_right_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        res = self.right_tracker.update(self.latest_right_image, rospy.get_time())
        if res is not None: 
            self.sync_best_sign(res, self.right_tracker.max_sign_area)

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
                self.angular_nudge = 1.0
            else:
                lin_x, ang_z = self.brains[self.current_brain_index].get_command(cv_image)
                self.publish_velocity(lin_x * self.velocity_nudge, ang_z * self.angular_nudge)

        # --- STATE: PAUSED ---
        elif self.current_state == RobotState.PAUSED:
            self.publish_velocity(0.0, 0.0)
            
            if self.is_movement_detected:
                rospy.loginfo("Movement detected! Resetting timer.")
                self.pause_start_time = current_time
                self.ears.reset_baseline()
                self.is_movement_detected = False
                
            elapsed = (current_time - self.pause_start_time).to_sec()
            
            if elapsed >= 2.0:
                self.current_brain_index = (self.current_brain_index + 1) % len(self.brains)
                rospy.loginfo(f"All clear. Switching to Model {self.current_brain_index + 1}.")
                self.current_state = RobotState.DRIVING
                self.cooldown_finish_time = current_time

    def render_loop(self):
        rate = rospy.Rate(30) # Refresh the GUI at 30Hz
        while not rospy.is_shutdown():
            now = rospy.get_time()

            if (now - self.global_start_time) > self.max_timeout_duration and self.timeout_triggered is False:
                rospy.logwarn("Global timeout reached! Executing timeout protocol.")
                last_msg = "Team13,multi21,-1,NA"
                self.score_publisher.publish(last_msg) 
                self.timeout_triggered = True
            
            # 2. Process best sign logic
            if self.active_sign['image'] is not None:
                time_since_lock = now - self.active_sign['timestamp']

                # --- TRIGGER POINT ---
                if time_since_lock > self.comparison_window and not self.sign_processed:
                    rospy.loginfo("--- WINDOW CLOSED: Extracting & Reading ---")
                    try:
                        top_line, bottom_line, closed_mask = extract_and_sort_letters(self.active_sign['image'])
                        
                        topic_string = self.cnn.predict_line(closed_mask, top_line)
                        clue_string = self.cnn.predict_line(closed_mask, bottom_line)

                        rospy.loginfo(f">>> TOPIC: {topic_string}")
                        rospy.loginfo(f">>> CLUE:  {clue_string}")

                        sign_id = self.topic_map.get(topic_string, 9) 

                        msg = f"Team13,multi21,{sign_id},{clue_string}"
                        self.score_publisher.publish(msg) 

                        if sign_id == 8:
                            rospy.logwarn("Final sign read. Stopping!")
                            last_msg = "Team13,multi21,-1,NA"
                            self.score_publisher.publish(last_msg) 

                    except Exception as e:
                        # Ensures the bot doesn't crash if it looks at a distorted/bad sign
                        rospy.logwarn(f"Failed to read sign properly: {e}")
                    
                    self.sign_processed = True

                # --- DISPLAY LOGIC ---
                if time_since_lock < 3.0:
                    # cv2.imshow("Best Sign Found", self.active_sign['image'])
                    self.sign_window_open = True
                else:
                    self.active_sign['image'] = None
                    self.sign_processed = False 
                    if self.sign_window_open:
                        try:
                            cv2.destroyWindow("Best Sign Found")
                        except cv2.error:
                            pass 
                        self.sign_window_open = False

            cv2.waitKey(1)
            rate.sleep()

if __name__ == '__main__':
    try:
        controller = StateMachineController()
        controller.render_loop()
    except rospy.ROSInterruptException:
        pass