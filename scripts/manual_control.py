#!/usr/bin/env python3

import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, Joy
from geometry_msgs.msg import Twist
from sign_extract import SignTracker
from letter_extract import extract_and_sort_letters
from ocr import SignReaderCNN

# source ~/ros_ws/devel/setup.bash

# cd ~/ros_ws/src/2025_competition/enph353/enph353_utils/scripts

# ./run_sim.sh -vpgw

# rosrun joy joy_node _dev:=/dev/input/js2 _autorepeat_rate:=20 _deadzone:=0.0

# ./score_tracker.py

class ManualController:
    def __init__(self):
        rospy.init_node('manual_control_node')
        self.bridge = CvBridge()
        self.min_sign_size = 30000
        # Color Config
        self.LOWER_B, self.UPPER_B = (115, 125, 100), (125, 255, 210)
        self.LOWER_W, self.UPPER_W = (0, 0, 100), (3, 3, 255)

        self.sign_window_open = False

        model_path = "/home/fizzer/cnn_trainer/wei_dynasty.tflite"
        self.cnn = SignReaderCNN(model_path)
        
        # Single Source of Truth for the "Winner" sign
        self.active_sign = {'image': None, 'area': 0, 'timestamp': 0.0}

        # Trackers
        self.left_tracker = SignTracker(self.LOWER_B, self.UPPER_B, self.LOWER_W, self.UPPER_W, self.min_sign_size, 3.0)
        self.right_tracker = SignTracker(self.LOWER_B, self.UPPER_B, self.LOWER_W, self.UPPER_W, self.min_sign_size, 3.0)

        # Image cache for GUI
        self.feeds = {'front': None, 'left': None, 'right': None}
        self.comparison_window = 1.0  # Wait 1s for the 2nd camera to finish
        self.sign_processed = False   # Flag to ensure we only process a sign once
        self.sign_window_open = False

        # ROS Pubs/Subs
        self.pub = rospy.Publisher('/B1/cmd_vel', Twist, queue_size=1)
        rospy.Subscriber('/joy', Joy, self.joy_callback)
        rospy.Subscriber('/B1/rrbot/camera_top/image_raw', Image, self.front_cb)
        rospy.Subscriber('/B1/rrbot/camera_left/image_raw', Image, self.left_cb, queue_size=1, buff_size=2**24)
        rospy.Subscriber('/B1/rrbot/camera_right/image_raw', Image, self.right_cb, queue_size=1, buff_size=2**24)

    def sync_best_sign(self, img, area):
        now = rospy.get_time()
        time_since_last = now - self.active_sign['timestamp']
        
        # 1. NEW SIGN: It's been a long time (3s+) since we saw anything.
        if time_since_last > 3.0:
            self.active_sign = {'image': img, 'area': area, 'timestamp': now}
            rospy.loginfo("First camera locked a sign. Comparison window OPEN.")

        # 2. COMPARISON: We are within the 1.0s window of the first detection.
        elif time_since_last < self.comparison_window:
            if area > self.active_sign['area']:
                self.active_sign['image'] = img
                self.active_sign['area'] = area
                rospy.loginfo(f"Found a better view! New Area: {area}")

    def left_cb(self, data):
        self.feeds['left'] = self.bridge.imgmsg_to_cv2(data, "bgr8")
        res = self.left_tracker.update(self.feeds['left'], rospy.get_time())
        
        # CHANGE THIS LINE:
        if res is not None: 
            self.sync_best_sign(res, self.left_tracker.max_sign_area)

    def right_cb(self, data):
        self.feeds['right'] = self.bridge.imgmsg_to_cv2(data, "bgr8")
        res = self.right_tracker.update(self.feeds['right'], rospy.get_time())
        
        # CHANGE THIS LINE:
        if res is not None: 
            self.sync_best_sign(res, self.right_tracker.max_sign_area)

    def front_cb(self, data):
        self.feeds['front'] = self.bridge.imgmsg_to_cv2(data, "bgr8")

    def joy_callback(self, data):
        t = Twist()
        lin, ang = data.axes[1], data.axes[3]
        t.linear.x = lin * 0.5 if abs(lin) > 0.05 else 0.0
        t.angular.z = ang * 2.25 if abs(ang) > 0.05 else 0.0
        self.pub.publish(t)

    def render_loop(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            now = rospy.get_time()
            
            # 1. Show the raw camera feeds
            # if self.feeds['front'] is not None: cv2.imshow("Front", self.feeds['front'])
            # if self.feeds['left'] is not None: cv2.imshow("Left", self.feeds['left'])
            # if self.feeds['right'] is not None: cv2.imshow("Right", self.feeds['right'])

            # 2. Logic for handling the "Best Sign"
            if self.active_sign['image'] is not None:
                time_since_lock = now - self.active_sign['timestamp']

                # --- TRIGGER POINT: Pick the winner after the window closes ---
                if time_since_lock > self.comparison_window and not self.sign_processed:
                    rospy.loginfo("--- WINDOW CLOSED: Extracting & Reading ---")
                    
                    # Note we are unpacking 3 things now!
                    top_line, bottom_line, closed_mask = extract_and_sort_letters(self.active_sign['image'])
                    
                    # --- NEW: Feed to CNN ---
                    topic_string = self.cnn.predict_line(closed_mask, top_line)
                    clue_string = self.cnn.predict_line(closed_mask, bottom_line)

                    rospy.loginfo(f">>> TOPIC: {topic_string}")
                    rospy.loginfo(f">>> CLUE:  {clue_string}")

                    # (You can keep your cv2.rectangle drawing code here for debugging)
                    self.sign_processed = True

                # --- DISPLAY LOGIC: Keep the winner on screen for 3 seconds ---
                if time_since_lock < 3.0:
                    # cv2.imshow("Best Sign Found", self.active_sign['image'])
                    self.sign_window_open = True
                else:
                    # Time is up! Flush the sign so the robot can look for the next one
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
        ManualController().render_loop()
    except rospy.ROSInterruptException: pass