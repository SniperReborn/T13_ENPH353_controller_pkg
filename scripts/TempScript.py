#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import math
from enum import Enum, auto

class State(Enum):
    FOLLOW_LINE = auto()
    ROUNDABOUT = auto()
    WAITING = auto()

class B1Controller:
    def __init__(self):
        rospy.init_node('linefollow')
        self.bridge = CvBridge()

        # Initially start with line following
        self.state = State.FOLLOW_LINE
        self.current_lane_width = 400
        self.last_road_state = "UNKNOWN"
        self.road_state_start_time = rospy.get_time()
        self.steering_bias = 0.0 
        self.state_start_time = rospy.get_time()

        # Object detection
        self.min_dist_front = 10.0
        self.obstacle_detected = False

        # Intersection
        self.intersection_count = 0
        self.last_wait_exit_time = 0.0
        self.cooldown_period = 5.0
        
        # PID Gains
        self.kP = 0.02
        self.kD = 0.003
        
        # Error tracking & Memory
        self.last_error = 0.0
        self.last_raw_line_error = 0.0  # Added to prevent derivative kick
        self.last_time = rospy.get_time()
        self.base_speed = 1
        self.alpha = 0.3
        
        # Memory variables for lane boundaries
        self.last_left_x = 200
        self.last_right_x = 600

        # Threshold path HSV
        self.lower_path = np.array([0, 0, 0])
        self.upper_path = np.array([0, 0, 95])
        # Threshold side HSV
        self.lower_side = np.array([0, 0, 200])
        self.upper_side = np.array([0, 0, 255])

        # ROS Pubs/Subs
        self.pub = rospy.Publisher('/B1/cmd_vel', Twist, queue_size=1)
        self.sub = rospy.Subscriber('/B1/rrbot/camera1/image_raw', Image, self.callback)
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        
        rospy.on_shutdown(self.shutdown)

    def scan_callback(self, msg):
        center_idx = len(msg.ranges) // 2
        front_slice = msg.ranges[center_idx-40 : center_idx+40]
        
        # Filter out inf and nan values to prevent min() from crashing
        valid_ranges = [r for r in front_slice if not math.isinf(r) and not math.isnan(r)]
        if valid_ranges:
            self.min_dist_front = min(valid_ranges)
        else:
            self.min_dist_front = 10.0
    
        self.truck_detected = self.min_dist_front < 5.0

    def callback(self, data):
        # Extract camera feed
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return
        
        h, w, _ = cv_image.shape
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        blur = cv2.GaussianBlur(hsv, (15,15), 25)
        
        lane_mask = cv2.inRange(blur, self.lower_side, self.upper_side)
        path_mask = cv2.inRange(blur, self.lower_path, self.upper_path)
        
        kernel = np.ones((7,7), np.uint8)
        lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, kernel)
        path_mask = cv2.morphologyEx(path_mask, cv2.MORPH_CLOSE, kernel)

        mid_crop = path_mask[int(0.5*h):int(0.75*h), :]
        bot_crop = lane_mask[int(0.75*h):int(h), :]
        cv2.imshow("Path view", mid_crop)

        current_road_state = self.classify_road(path_mask, h, w)
        rospy.loginfo_throttle(1.0, f"Road state: {current_road_state}")
        now = rospy.get_time()
        time_since_last_wait = now - self.last_wait_exit_time

        if current_road_state != self.last_road_state:
            self.road_state_start_time = now
            self.last_road_state = current_road_state

        duration = now - self.road_state_start_time

        if self.state == State.FOLLOW_LINE:
            if current_road_state == "T-STRAIGHT" and duration > 0.1 and self.intersection_count < 2:
                if time_since_last_wait > self.cooldown_period:
                    rospy.loginfo(f"T-Intersection {self.intersection_count + 1} detected. Stopping...")
                    self.transition_to(State.WAITING)
                    return
                else:
                    rospy.loginfo_throttle(1, "Ignoring T-Straight (Cooldown active)")

            if current_road_state == "T-STRAIGHT" and duration > 0.25:
                self.steering_bias = 60  
            elif current_road_state == "T-LEFT TURN" and duration > 0.15:
                self.steering_bias = 100  
            else:
                self.steering_bias = 0.0

            self.follow_line(bot_crop, w)

        elif self.state == State.WAITING:
            self.handle_truck_scan()
            return       

        cv2.waitKey(1)

    def handle_truck_scan(self):
        elapsed = rospy.get_time() - self.state_start_time
        move = Twist() 
        
        if elapsed < 3.0:
            self.pub.publish(move)
        else:
            if self.truck_detected:
                self.pub.publish(move)
            else:
                self.intersection_count += 1
                rospy.loginfo(f"Path clear! Intersections cleared: {self.intersection_count}/2")
                self.last_wait_exit_time = rospy.get_time() 
                self.steering_bias = 60 
                self.transition_to(State.FOLLOW_LINE)
    
    def transition_to(self, next_state):
        if self.state == next_state:
            return

        rospy.loginfo(f"--- STATE CHANGE: {self.state.name} -> {next_state.name} ---")
        self.state = next_state
        self.state_start_time = rospy.get_time()

        if next_state == State.FOLLOW_LINE:
            self.last_error = 0.0
            self.last_raw_line_error = 0.0

    def classify_road(self, road_mask, h, w):
        top_win = road_mask[int(0.5*h):int(0.6*h), :]
        mid_win = road_mask[int(0.6*h):int(0.7*h), :]
        bot_win = road_mask[int(0.8*h):int(h), :]
        t_cx, t_w = self.get_stats(top_win)
        m_cx, m_w = self.get_stats(mid_win)
        b_cx, b_w = self.get_stats(bot_win)

        turn_sensitivity = 1.1
        width_sensitivity_straight = 0.9
        width_sensitivity = 0.3

        if m_cx is None or b_cx is None or t_cx is None:
            return "UNKNOWN"

        if m_cx > turn_sensitivity * b_cx:
            if m_w > b_w * width_sensitivity and t_cx < m_cx:
                return "T-RIGHT TURN"
            else:
                return "RIGHT TURN"
        elif turn_sensitivity * m_cx < b_cx:
            if m_w > b_w * width_sensitivity and t_cx > m_cx:
                return "T-LEFT TURN"
            else:
                return "LEFT TURN"
        else:
            if m_w > b_w * width_sensitivity_straight:
                return "T-STRAIGHT"
            else:
                return "STRAIGHT"

    def get_stats(self, window):
        M = cv2.moments(window)
        if M['m00'] < 500: return None, 0
        cx = int(M['m10'] / M['m00'])
        cnts, _ = cv2.findContours(window, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return cx, 0
        _, _, w_rect, _ = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        return cx, w_rect

    def follow_line(self, mask, width):
        error = 0
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

        if len(centroids) > 2:
            # CROSSWALK DETECTED
            error = 0
            rospy.loginfo_throttle(0.5, "Crosswalk mode (Driving Straight)")
            mask = cv2.circle(mask, (int(width/2), int(mask.shape[0]/2)), 20, (255, 0, 0), -1)

        elif len(centroids) == 2:
            left_line = centroids[0]
            right_line = centroids[-1]
            span = abs(right_line[0] - left_line[0])

            # Update our memory of where the boundaries are
            self.last_left_x = left_line[0] 
            self.last_right_x = right_line[0]

            if span > 150:
                self.current_lane_width = (0.9 * self.current_lane_width) + (0.1 * span)
                lane_center = (left_line[0] + right_line[0]) / 2
                error = (width / 2) - lane_center
                mask = cv2.circle(mask, (left_line[0], left_line[1]), 20, (0, 255, 0), -1)
                mask = cv2.circle(mask, (right_line[0], right_line[1]), 20, (0, 255, 0), -1)
            else:
                line_x = (left_line[0] + right_line[0]) / 2 
                error = self.calculate_single_line_error(line_x, width)
                mask = cv2.circle(mask, (int(line_x), int(left_line[1])), 20, (0, 255, 0), -1)

        elif len(centroids) == 1:
            line_x = centroids[0][0]
            line_y = centroids[0][1]
            error = self.calculate_single_line_error(line_x, width)
            mask = cv2.circle(mask, (line_x, line_y), 20, (0, 255, 0), -1)

        else:
            # BLIND SPOT
            rospy.logwarn_throttle(0.5, "LOST LINE! Braking...")
            error = self.last_raw_line_error
        
        cv2.imshow("Lane View", mask)

        # --- MATH & PID CALCULATION SECTION ---
        max_error = width / 2.0
        current_time = rospy.get_time()
        dt = current_time - self.last_time
        
        # Calculate derivative purely on the physical line location to prevent spikes
        if dt > 0.01: 
            deriv = (error - self.last_raw_line_error) / dt
        else:
            deriv = 0

        # Now apply the bias and smoothing for the Proportional term
        raw_error = error + self.steering_bias  
        self.smoothed_error = (self.alpha * raw_error) + (1.0 - self.alpha) * self.last_error

        prop_term = self.kP * self.smoothed_error
        deriv_term = self.kD * deriv

        # --- SPEED SCALING SECTION ---
        if len(centroids) == 0:
            # Drop speed heavily if we are driving blind
            current_speed = self.base_speed * 0.3
        else:
            speed_factor = 1.0 - (min(abs(raw_error), max_error) / max_error) * 0.85
            current_speed = self.base_speed * speed_factor

        # Publish the command
        move = Twist()
        move.linear.x = current_speed
        move.angular.z = float(prop_term + deriv_term)
        self.pub.publish(move)

        # Store states for the next loop
        self.last_time = current_time
        self.last_error = self.smoothed_error
        self.last_raw_line_error = error
    
    def calculate_single_line_error(self, line_x, image_width):
        half_w = self.current_lane_width / 2.0
        
        # Compare current line to our memory
        dist_to_left = abs(line_x - self.last_left_x)
        dist_to_right = abs(line_x - self.last_right_x)

        if dist_to_left < dist_to_right:
            # It's the LEFT line. Target is to the RIGHT of it.
            target_x = line_x + half_w
            self.last_left_x = line_x 
        else:
            # It's the RIGHT line. Target is to the LEFT of it.
            target_x = line_x - half_w
            self.last_right_x = line_x
              
        return (image_width / 2.0) - target_x

    def shutdown(self):
        self.pub.publish(Twist())
        rospy.loginfo("Robot Stopped.")

if __name__ == '__main__':
    bot = B1Controller()
    rospy.spin()