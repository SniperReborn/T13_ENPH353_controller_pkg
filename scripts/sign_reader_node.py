#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2

# Sign Reading Imports
from sign_extract import SignTracker
from letter_extract import extract_and_sort_letters
from ocr import SignReaderCNN

class SignReaderNode:
    def __init__(self):
        rospy.init_node('sign_reader_node')
        self.bridge = CvBridge()
        
        # --- Sign Reading Components ---
        self.min_sign_size = 20000
        self.LOWER_B, self.UPPER_B = (115, 125, 100), (125, 255, 210)
        self.LOWER_W, self.UPPER_W = (0, 0, 100), (3, 3, 255)

        self.cnn = SignReaderCNN("/home/fizzer/cnn_trainer/wei_dynasty.tflite")
        
        self.active_sign = {'image': None, 'area': 0, 'timestamp': 0.0}
        self.left_tracker = SignTracker(self.LOWER_B, self.UPPER_B, self.LOWER_W, self.UPPER_W, self.min_sign_size, 3.0)
        self.right_tracker = SignTracker(self.LOWER_B, self.UPPER_B, self.LOWER_W, self.UPPER_W, self.min_sign_size, 3.0)

        self.comparison_window = 1.0  
        self.sign_processed = False   

        self.topic_map = {
            "SIZE": 1, "VICTIM": 2, "CRIME": 3, "TIME": 4, 
            "PLACE": 5, "MOTIVE": 6, "WEAPON": 7, "BANDIT": 8
        }

        # Publisher for the driving node
        self.detected_sign_pub = rospy.Publisher("/B1/detected_signs", String, queue_size=5)

        # Raw image subscribers with massive buffers
        self.left_sub = rospy.Subscriber('/B1/rrbot/camera_left/image_raw', Image, self.left_callback, queue_size=1, buff_size=2**24)
        self.right_sub = rospy.Subscriber('/B1/rrbot/camera_right/image_raw', Image, self.right_callback, queue_size=1, buff_size=2**24)
        
        rospy.loginfo("Sign Reader Node Initialized. Watching for signs...")

    def sync_best_sign(self, img, area):
        now = rospy.get_time()
        time_since_last = now - self.active_sign['timestamp']
        
        if time_since_last > 3.0:
            self.active_sign = {'image': img, 'area': area, 'timestamp': now}
            rospy.loginfo("Locked a sign. Comparison window OPEN.")
        elif time_since_last < self.comparison_window:
            if area > self.active_sign['area']:
                self.active_sign['image'] = img
                self.active_sign['area'] = area

    def left_callback(self, data):
        cv_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        res = self.left_tracker.update(cv_img, rospy.get_time())
        if res is not None: self.sync_best_sign(res, self.left_tracker.max_sign_area)

    def right_callback(self, data):
        cv_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        res = self.right_tracker.update(cv_img, rospy.get_time())
        if res is not None: self.sync_best_sign(res, self.right_tracker.max_sign_area)

    def process_loop(self):
        rate = rospy.Rate(10) # 10Hz checking loop
        while not rospy.is_shutdown():
            if self.active_sign['image'] is not None:
                time_since_lock = rospy.get_time() - self.active_sign['timestamp']

                if time_since_lock > self.comparison_window and not self.sign_processed:
                    rospy.loginfo("Extracting & Reading Sign...")
                    try:
                        top_line, bottom_line, closed_mask = extract_and_sort_letters(self.active_sign['image'])
                        topic_string = self.cnn.predict_line(closed_mask, top_line)
                        clue_string = self.cnn.predict_line(closed_mask, bottom_line)

                        sign_id = self.topic_map.get(topic_string, 9) 
                        
                        # Pack the data and send it to the driver node
                        msg = String()
                        msg.data = f"{sign_id},{clue_string}"
                        self.detected_sign_pub.publish(msg)
                        rospy.loginfo(f"Published Sign -> ID: {sign_id}, Clue: {clue_string}")

                    except Exception as e:
                        rospy.logwarn(f"Failed to read sign: {e}")
                    
                    self.sign_processed = True

                if time_since_lock >= 3.0:
                    self.active_sign['image'] = None
                    self.sign_processed = False 

            rate.sleep()

if __name__ == '__main__':
    try:
        reader = SignReaderNode()
        reader.process_loop()
    except rospy.ROSInterruptException:
        pass