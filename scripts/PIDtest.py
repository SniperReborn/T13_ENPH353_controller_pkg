#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
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
        self.last_error = 0.0
        self.last_time = rospy.get_time()
        self.base_speed = 1
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
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        
        rospy.on_shutdown(self.shutdown)

    def scan_callback(self, msg):
        # LaserScan.ranges is an array. 
        # For a 180 deg scan with 720 samples, index 360 is dead ahead.
        # We check a slice of 40 degrees in front
        center_idx = len(msg.ranges) // 2
        front_slice = msg.ranges[center_idx-40 : center_idx+40]
        self.min_dist_front = min(front_slice)
    
        # Threshold for truck (e.g., something within 1.5 meters)
        self.truck_detected = self.min_dist_front < 1.5

    def callback(self, data):
        # Extract camera feed
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return
        # Extract image properties
        h, w, _ = cv_image.shape

        # Convert to HSV to remove sensitivity to shadow
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Blur image to smooth over small gaps
        blur = cv2.GaussianBlur(hsv, (15,15), 25)
        
        # Processing the Mask
        lane_mask = cv2.inRange(blur, self.lower_side, self.upper_side)
        path_mask = cv2.inRange(blur, self.lower_path, self.upper_path)
        
        # Closing small holes in the black and white mask
        kernel = np.ones((7,7), np.uint8)
        lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, kernel)
        path_mask = cv2.morphologyEx(path_mask, cv2.MORPH_CLOSE, kernel)

        # Cropping
        mid_crop = path_mask[int(0.5*h):int(0.75*h), :]
        bot_crop = lane_mask[int(0.75*h):int(h), :]
        cv2.imshow("Path view", mid_crop)

        # Road state processing
        current_road_state = self.classify_road(path_mask, h, w)
        rospy.loginfo(f"Road state: {current_road_state}")
        now = rospy.get_time()
        time_since_last_wait = now - self.last_wait_exit_time

        # Tracking road state duration
        if current_road_state != self.last_road_state:
            self.road_state_start_time = now
            self.last_road_state = current_road_state

        duration = now - self.road_state_start_time

        if self.state == State.FOLLOW_LINE:
            # 1. STOPPING LOGIC (Now capped at 2 stops)
            if current_road_state == "T-STRAIGHT" and duration > 0.1 and self.intersection_count < 2:
                if time_since_last_wait > self.cooldown_period:
                    rospy.loginfo(f"T-Intersection {self.intersection_count + 1} detected. Stopping...")
                    self.transition_to(State.WAITING)
                    return
                else:
                    rospy.loginfo_throttle(1, "Ignoring T-Straight (Cooldown active)")

            # 2. BIASING LOGIC (Runs indefinitely so you can always make turns)
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
                # --- INCREMENT COUNTER AND APPLY COOLDOWN ---
                self.intersection_count += 1
                rospy.loginfo(f"Path clear! Intersections cleared: {self.intersection_count}/2")
                
                self.last_wait_exit_time = rospy.get_time() 
                self.steering_bias = 60 
                self.transition_to(State.FOLLOW_LINE)
    
    def transition_to(self, next_state):
        """
        Handles the hand-off between states. 
        Ensures logs are kept and timers are reset.
        """
        # 1. Check if we are already in that state (prevents redundant resets)
        if self.state == next_state:
            return

        # 2. Log the change for debugging (This is a lifesaver in ROS)
        rospy.loginfo(f"--- STATE CHANGE: {self.state.name} -> {next_state.name} ---")

        # 3. Update the state
        self.state = next_state

        # 4. Reset the internal clock
        # This allows 'duration' in the next state to start from 0.0
        self.state_start_time = rospy.get_time()

        # 5. (Optional) State-specific resets
        # For example, if entering FOLLOW_LINE, we might want to clear the old PID error
        if next_state == State.FOLLOW_LINE:
            self.last_error = 0.0

    def classify_road(self, road_mask, h, w):
        top_win = road_mask[int(0.5*h):int(0.6*h), :]
        mid_win = road_mask[int(0.6*h):int(0.7*h), :]
        bot_win = road_mask[int(0.8*h):int(h), :]
        t_cx, t_w = self.get_stats(top_win)
        m_cx, m_w = self.get_stats(mid_win)
        b_cx, b_w = self.get_stats(bot_win)

        turn_sensitivity = 1.15
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
        if M['m00'] < 500: return None, 0 # Too little road visible
        cx = int(M['m10'] / M['m00'])
        # Use a bounding box to get the 'width' of the road in this window
        cnts, _ = cv2.findContours(window, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return cx, 0
        _, _, w_rect, _ = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        return cx, w_rect

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

            if span > 150:
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
        
        cv2.imshow("Lane View", mask)

        max_error = width / 2.0
        raw_error = error + self.steering_bias  # Keep the raw pixel error for linear calculations

        # Display current raw error
        rospy.loginfo(f"Error = {raw_error}")

        # Error smoothing
        self.smoothed_error = (self.alpha * raw_error) + (1.0 - self.alpha) * self.last_error

        # Calculate time step for derivative calculation
        current_time = rospy.get_time()
        dt = current_time - self.last_time
        if dt > 0.01: 
            deriv = (self.smoothed_error - self.last_error) / dt
        else:
            deriv = 0

        # Basic PD tuning
        prop_term = self.kP * self.smoothed_error
        deriv_term = self.kD * deriv

        # Alter the speed depending on how big the error is. The bigger the error, the slower the vehicle.
        speed_factor = 1.0 - (min(abs(raw_error), max_error) / max_error) * 0.85
        current_speed = self.base_speed * speed_factor

        move = Twist()
        move.linear.x = current_speed
        move.angular.z = float(prop_term + deriv_term)

        self.pub.publish(move)

        # Store last time stamp and the smoothed error
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