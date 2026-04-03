#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from enum import Enum, auto

class State(Enum):
    FOLLOW_LINE = auto()
    ROUNDABOUT = auto()
    STOPPED = auto()

class B1Controller:
    def __init__(self):
        rospy.init_node('linefollow')
        self.bridge = CvBridge()
        self.state = State.FOLLOW_LINE
        self.current_lane_width = 400
        
        # PID Gains
        self.kP = 0.02
        self.kD = 0.002
        self.last_error = 0.0
        self.last_time = rospy.get_time()
        self.base_speed = 0.75 # Dropped slightly for stability during testing
        self.alpha = 0.3

        # Threshold path HSV
        self.lower_path = np.array([0, 0, 0])
        self.upper_path = np.array([0, 0, 95])
        # Threshold side HSV
        self.lower_side = np.array([0, 0, 200])
        self.upper_side = np.array([0, 0, 255])

        # ROS Pubs/Subs
        self.pub = rospy.Publisher('/B1/cmd_vel', Twist, queue_size=1)
        self.sub = rospy.Subscriber('/B1/rrbot/camera1/image_raw', Image, self.callback)
        
        rospy.on_shutdown(self.shutdown)

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        h, w, _ = cv_image.shape

        bot_crop = cv_image[int(0.75*h):int(h), :]
        cv2.imshow("cropped view", bot_crop)

        hsv = cv2.cvtColor(bot_crop, cv2.COLOR_BGR2HSV)

        blur = cv2.GaussianBlur(hsv, (15,15), 25)

        #cv2.imshow("blurred view", blur)
        
        # Processing the Mask
        path_mask = cv2.inRange(blur, self.lower_side, self.upper_side)
        
        # ROI Cropping (Staged windows)
        # Bottom window for steering, Top window for "event detection"
        kernel = np.ones((7,7), np.uint8)
        path_mask = cv2.morphologyEx(path_mask, cv2.MORPH_CLOSE, kernel)

        self.follow_line(path_mask, w)

        #cv2.imshow("Mask Path View", path_mask)

        cv2.waitKey(1)

    def follow_line(self, mask, width):
        # Initialize variables
        error = 0
        cx = 0
        cy = 0

        # Detect contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sort through contours
        centroids = []
        for contour in contours:
            if cv2.contourArea(contour) > 200:
                M = cv2.moments(contour, binaryImage = True)
                if M['m00'] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    centroids.append((cx, cy))

        centroids.sort(key=lambda x: x[0])
        
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        if len(centroids) >= 2:
            left_line = centroids[0]
            right_line = centroids[-1]
            span = abs(right_line[0] - left_line[0])

            if span > 60:
                # VALID LANE: We see both sides
                self.current_lane_width = (0.9 * self.current_lane_width) + (0.1 * span)
                lane_center = (left_line[0] + right_line[0]) / 2
                error = (width / 2) - lane_center
                rospy.loginfo("Centering mode (Double Line)")

                mask = cv2.circle(mask, (left_line[0], left_line[1]), 20, (0, 255, 0), -1)
                mask = cv2.circle(mask, (right_line[0], right_line[1]), 20, (0, 255, 0), -1)
            else:
                # BROKEN LINE: These are just segments of one side
                # Treat this exactly like the 'len(centroids) == 1' case
                line_x = (left_line[0] + right_line[0]) / 2 # Average them for a better estimate
                error = self.calculate_single_line_error(line_x, width)
                rospy.loginfo("Line following mode (Broken Single Line)")

                mask = cv2.circle(mask, (right_line[0], right_line[1]), 20, (0, 255, 0), -1)
            
            rospy.loginfo(f"Lane width = {self.current_lane_width}")

        elif len(centroids) == 1:
            line_x = centroids[0][0]
            line_y = centroids[0][1]
            error = self.calculate_single_line_error(line_x, width)
            rospy.loginfo("Line following mode")
            rospy.loginfo(f"Lane width = {self.current_lane_width}")

            mask = cv2.circle(mask, (line_x, line_y), 20, (0, 255, 0), -1)

        else:
            error = self.last_error
        
        cv2.imshow("Path View", mask)

        max_error = width / 2.0
        raw_error = error  # Keep the raw pixel error for linear calculations

        rospy.loginfo(f"Error = {raw_error}")

        self.smoothed_error = (self.alpha * raw_error) + (1.0 - self.alpha) * self.last_error

        current_time = rospy.get_time()
        dt = current_time - self.last_time

        if dt > 0.01: 
            deriv = (self.smoothed_error - self.last_error) / dt
        else:
            deriv = 0

        prop_term = self.kP * self.smoothed_error
        deriv_term = self.kD * deriv

        speed_factor = 1.0 - (min(abs(raw_error), max_error) / max_error) * 0.85
        current_speed = self.base_speed * speed_factor

        move = Twist()
        move.linear.x = current_speed
        move.angular.z = float(prop_term + deriv_term)

        self.pub.publish(move)
        self.last_time = current_time
        self.last_error = self.smoothed_error
    
    def calculate_single_line_error(self, line_x, image_width):
        # Use your 'memory' of the lane width
        half_w = self.current_lane_width / 2
        
        if line_x < (image_width / 2):
            # It's the LEFT line. Target is to the RIGHT of it.
            target_x = line_x + half_w
        else:
            # It's the RIGHT line. Target is to the LEFT of it.
            target_x = line_x - half_w
              
        return (image_width / 2) - target_x

    def shutdown(self):
        self.pub.publish(Twist())
        rospy.loginfo("Robot Stopped.")

if __name__ == '__main__':
    bot = B1Controller()
    rospy.spin()