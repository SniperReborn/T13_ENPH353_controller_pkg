#!/usr/bin/env python3
import cv2
import numpy as np

class PinkDetector:
    def __init__(self):
        # HSV Color range for Pink/Magenta in Gazebo
        # You may need to tweak these if the lighting is weird
        self.lower_pink = np.array([147, 250, 250])
        self.upper_pink = np.array([170, 255, 255])
        
        # How many pink pixels we need to see to trigger a "stop"
        self.trigger_threshold = 500

    def is_pink_line_present(self, cv_image):
        h, w = cv_image.shape[:2]
        
        # Only look at the bottom 40% of the screen so we don't trigger on background objects
        bottom_strip = cv_image[int(h * 0.6):h, :]
        
        # Convert to HSV and create a mask
        hsv = cv2.cvtColor(bottom_strip, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_pink, self.upper_pink)
        
        # Count the white pixels in the mask
        pink_pixel_count = cv2.countNonZero(mask)
        
        return pink_pixel_count > self.trigger_threshold