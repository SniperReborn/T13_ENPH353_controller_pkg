#!/usr/bin/env python3
import cv2
import numpy as np
import rospy

def _order_points(pts):
    """Sorts 4 points: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def extract_and_warp(cv_image, lower_border, upper_border, lower_white, upper_white, min_area=3500):
    """Finds blue border -> Warps 300x200 -> Finds white inner box -> Crops."""
    hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower_border), np.array(upper_border))
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None, 0
    
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    
    if area < min_area: return None, 0
        
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    
    if len(approx) == 4:
        pts = approx.reshape(4, 2)
        rect = _order_points(pts)
        
        # Warp to fixed 300x200
        tw, th = 300, 200
        dst = np.array([[0, 0], [tw-1, 0], [tw-1, th-1], [0, th-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(cv_image, M, (tw, th))
        
        # Extract White Inner Box
        warped_hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(warped_hsv, np.array(lower_white), np.array(upper_white))
        w_contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if w_contours:
            best_w = max(w_contours, key=cv2.contourArea)
            if cv2.contourArea(best_w) > (area * 0.1): # Ensure it's not just noise
                x, y, w, h = cv2.boundingRect(best_w)
                # Crop with a tiny margin
                crop = warped[max(0,y-5):min(th,y+h+5), max(0,x-5):min(tw,x+w+5)]
                return crop, area

    return None, 0

class SignTracker:
    def __init__(self, lower_border, upper_border, lower_white, upper_white, min_area=3500, cooldown=3.0):
        self.params = (lower_border, upper_border, lower_white, upper_white)
        self.min_area = min_area
        self.cooldown = cooldown
        self.last_proc_time = 0.0
        self.is_tracking = False
        self.max_sign_area = 0
        self.best_sign_image = None

    def update(self, cv_image, now):
        if (now - self.last_proc_time) <= self.cooldown or cv_image is None:
            return None

        try:
            flat_sign, area = extract_and_warp(cv_image, *self.params, self.min_area)
        except Exception as e:
            rospy.logwarn(f"Extraction error: {e}")
            flat_sign, area = None, 0

        if flat_sign is not None:
            self.is_tracking = True
            if area > self.max_sign_area:
                self.max_sign_area = area
                self.best_sign_image = flat_sign
            return None
        elif self.is_tracking:
            # Sign just left the view
            res = self.best_sign_image
            self.last_proc_time = now
            self.is_tracking = False
            self.max_sign_area = 0
            self.best_sign_image = None
            return res
        return None