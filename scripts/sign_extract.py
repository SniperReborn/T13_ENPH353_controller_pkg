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

def extract_and_warp(cv_image, lower_border, upper_border, lower_white, upper_white, min_area):
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
        tw, th = 900, 600
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
                inset = 15 
                
                # We move the start (x, y) INWARD (+) and the end (x+w, y+h) INWARD (-)
                y_start = max(0, y + inset)
                y_end   = min(th, y + h - inset)
                x_start = max(0, x + inset)
                x_end   = min(tw, x + w - inset)
                
                # Safety check: Ensure we didn't inset so much that the image disappeared
                if y_end > y_start and x_end > x_start:
                    crop = warped[y_start:y_end, x_start:x_end]
                    crop = cv2.resize(crop, (600, 400))
                    return crop, area

    return None, 0

class SignTracker:
    def __init__(self, lower_border, upper_border, lower_white, upper_white, min_area, cooldown):
        self.params = (lower_border, upper_border, lower_white, upper_white)
        self.min_area = min_area
        self.cooldown = cooldown
        self.last_proc_time = 0.0
        self.is_tracking = False
        self.max_sign_area = 0
        self.best_sign_image = None
        
        # --- NEW: Patience variables ---
        self.time_lost = 0.0
        self.patience = 0.5  # Wait 0.5s before giving up on the sign

    def update(self, cv_image, now):
        # 1. Check Cooldown
        if (now - self.last_proc_time) <= self.cooldown or cv_image is None:
            return None

        # 2. Extract
        try:
            flat_sign, area = extract_and_warp(cv_image, *self.params, self.min_area)
        except Exception as e:
            rospy.logwarn(f"Extraction error: {e}")
            flat_sign, area = None, 0

        # 3. Handle Tracking Logic
        if flat_sign is not None:
            self.is_tracking = True
            self.time_lost = 0.0  # We see the sign! Reset the panic timer.
            
            # Keep updating the best image as we get closer
            if area > self.max_sign_area:
                self.max_sign_area = area
                self.best_sign_image = flat_sign
            return None
            
        elif self.is_tracking:
            # We lost the sign! Start the timer.
            if self.time_lost == 0.0:
                self.time_lost = now
                
            # If it's been gone for more than 0.5 seconds, it's ACTUALLY gone.
            if (now - self.time_lost) > self.patience:
                res = self.best_sign_image
                self.last_proc_time = now  # START 3-SECOND COOLDOWN NOW
                self.is_tracking = False
                self.max_sign_area = 0
                self.best_sign_image = None
                self.time_lost = 0.0
                return res
                
        return None