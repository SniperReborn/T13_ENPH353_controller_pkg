#! /usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from scipy.signal import find_peaks
from scipy.ndimage import label

bridge = CvBridge()

def order_pts(pts):
    pts = pts.reshape(4,2)

    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = pts[:,0] - pts[:,1]

    rect[1] = pts[np.argmax(diff)]
    rect[3] = pts[np.argmin(diff)]

    return rect

def perspective_transform(image, pts):
    rect = order_pts(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def board_detection(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_limit = (120, 50, 50)
    upper_limit = (130, 255, 255)
    mask = cv2.inRange(hsv_frame, lower_limit, upper_limit)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = []
    for cont in contours:
        area = cv2.contourArea(cont)
        if area > 22000:
            valid_contours.append(cont)

    rect_contours = []
    for cont in valid_contours:
        peri = cv2.arcLength(cont, True)
        approx = cv2.approxPolyDP(cont, 0.02*peri, True)

        _, _, w, h = cv2.boundingRect(cont)
        ratio = w/h

        if len(approx) == 4 and ratio > 0.9:
           rect_contours.append(approx)

    best_contour = None
    max_area = 0

    for cont in rect_contours:
        area = cv2.contourArea(cont)

        if area > max_area:
            max_area = area
            best_contour = cont

    if best_contour is not None:
        warped = perspective_transform(frame, best_contour)
        return warped, best_contour

    return None, None

def char_crop(board):
    if board is None:
        return []

    h, w, _ = board.shape

    value_region = board[h//2:int(h*0.90), :]
    hv, wv, _ = value_region.shape

    value_region = value_region[:, int(wv*0.05):int(wv*0.95)]
    value_region = cv2.resize(value_region, None, fx=2, fy=2)

    gray = cv2.cvtColor(value_region, cv2.COLOR_BGR2GRAY)

    bg_mean = np.mean(gray[-5:, :]) - 5

    dark_mask = gray < bg_mean
    col_count = np.sum(dark_mask, axis=0)

    col_mask = (col_count > 0).astype(int)

    transitions = np.diff(col_mask)

    starts = np.where(transitions == -1)[0] + 1
    ends = np.where(transitions == 1)[0] + 1

    if len(starts) > 0 and len(ends) > 0:
        left = starts[0]
        right = ends[-1]
        if left < right:
            col_count = col_count[left:right]
            value_region = value_region[:, left:right]

    col_mask = (col_count > 0).astype(int)

    zero_mask = (col_mask == 0).astype(int)
    labeled, _ = label(zero_mask)
    counts = np.bincount(labeled.ravel())

    remove_labels = np.where(counts >= 10)[0]
    remove_labels = remove_labels[remove_labels != 0]

    remove_mask = np.isin(labeled, remove_labels)
    keep = ~remove_mask

    col_count = col_count[keep]
    value_region = value_region[:, keep]

    cv2.imshow("value_region", value_region)

    smooth = np.convolve(col_count, np.ones(5)/5, mode='same')

    valleys, _ = find_peaks(-smooth, prominence=2, distance=20)

    ink_cols = np.where(col_count > 2)[0]
    if len(ink_cols) == 0:
        return []

    left = ink_cols[0]
    right = ink_cols[-1]

    ink_th = 2
    cut_points = [left]

    for v in valleys:
        if not (left < v < right):
            continue

        l = v
        while l > left and col_count[l] <= ink_th:
            l -= 1

        r = v
        while r < right and col_count[r] <= ink_th:
            r += 1

        if r - l > 3:
            cut_points.append(r)

    cut_points.append(right)
    cut_points = sorted(list(set(cut_points)))

    filtered_cut_points = [cut_points[0]]
    for p in cut_points[1:]:
        if p - filtered_cut_points[-1] > 5:
            filtered_cut_points.append(p)

    chars = []
    for i in range(len(filtered_cut_points) - 1):
        start = filtered_cut_points[i]
        end = filtered_cut_points[i + 1]

        if end - start > 5:
            char_img = value_region[:, start:end]
            chars.append(char_img)

    return chars

def process_frame(frame):
    cv_frame = bridge.imgmsg_to_cv2(frame, "bgr8")
    display_frame = cv_frame.copy()
    board, contour = board_detection(cv_frame)

    if contour is not None:
        cv2.drawContours(display_frame, [contour], -1, (0,0,255), 3)

    cv2.imshow("Clue Board Detection", display_frame)
    if board is not None:
        cv2.imshow("Perfect board", board)
        chars = char_crop(board)
        gap = np.zeros((chars[0].shape[0], 10, 3), dtype=np.uint8)

        combined = []
        for c in chars:
            combined.append(c)
            combined.append(gap)

        result = cv2.hconcat(combined[:-1])
        cv2.imshow("chars", result)
    else:
        cv2.imshow("Perfect board", np.zeros((100,100,3)))
    cv2.waitKey(1)

def listener():
    rospy.init_node("clue_board")
    rospy.Subscriber("/B1/rrbot/camera1/image_raw", Image, process_frame)
    rospy.spin()

if __name__ == "__main__":
    listener()
