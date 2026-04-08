import cv2
import numpy as np

def extract_and_sort_letters(img):
    """
    Takes a BGR image, isolates letters using morphology, 
    splits them into two lines, and sorts them left-to-right.
    """
    if img is None:
        return [], []

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Using the blue HSV bounds from your original script
    lower_blue = np.array([115, 75, 75])
    upper_blue = np.array([125, 255, 210])

    # Create a mask (Blue becomes white, everything else becomes black)
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = np.ones((5,5), np.uint8)
    eroded = cv2.erode(blue_mask, kernel, iterations=1)

    kernel_c = np.ones((3,3), np.uint8)
    closed = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, kernel_c)


    # cv2.imshow("Post-processed image", closed)
    
    # 4. Find Contours (The Letters)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # Filter out random noise
        if w > 5 and h > 10 and h < 150: 
            valid_boxes.append((x, y, w, h, c)) 
            
    if not valid_boxes:
        return [], []

    # --- NEW: DYNAMIC SPLITTING LOGIC ---
    # 1. Find the median width of all detected letters
    widths = [box[2] for box in valid_boxes] # box[2] is the width 'w'
    median_width = np.median(widths)
    
    split_boxes = []
    for box in valid_boxes:
        x, y, w, h, c = box
        
        # 2. Check if the box is abnormally wide (Threshold: 1.7x the median)
        if w > (median_width * 1.7):
            # Split the box in half
            w_half = w // 2
            
            # Left Half (Original X, Half Width)
            split_boxes.append((x, y, w_half, h, c))
            # Right Half (Shifted X, Remaining Width)
            split_boxes.append((x + w_half, y, w - w_half, h, c))
        else:
            # Normal letter, keep as is
            split_boxes.append(box)
            
    # Overwrite valid_boxes with our newly processed list
    valid_boxes = split_boxes
    # ------------------------------------

    # 5. Group by Y-Coordinate (Binary Distribution)
    # Calculate the exact center Y of every bounding box
    boxes_with_centers = [(box, box[1] + box[3]//2) for box in valid_boxes]
    
    # Find the average Y center of all letters combined. 
    avg_y = sum(center_y for _, center_y in boxes_with_centers) / len(boxes_with_centers)
    
    top_line = []
    bottom_line = []
    
    for box_tuple, center_y in boxes_with_centers:
        if center_y < avg_y:
            top_line.append(box_tuple)  # Above the average line
        else:
            bottom_line.append(box_tuple) # Below the average line
            
    # 6. Sort each array by X-Coordinate (Left to Right)
    top_line.sort(key=lambda b: b[0])
    bottom_line.sort(key=lambda b: b[0])
    
    return top_line, bottom_line, closed