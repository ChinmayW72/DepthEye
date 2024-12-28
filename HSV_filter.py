import cv2
import numpy as np

def add_HSV_filter(frame, camera):
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Define HSV ranges
    l_b_r = np.array([60, 110, 50])
    u_b_r = np.array([255, 255, 255])
    l_b_l = np.array([143, 110, 50])
    u_b_l = np.array([255, 255, 255])

    mask = cv2.inRange(hsv, l_b_r, u_b_r) if camera == 1 else cv2.inRange(hsv, l_b_l, u_b_l)

    # Erosion and dilation
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    return mask