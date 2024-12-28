import sys
import cv2
import numpy as np
import time
import imutils
from matplotlib import pyplot as plt

# Functions
import HSV_filter as hsv
import shape_recognition as shape
import triangulation as tri
# import calibration as calib

# Baseline, focal length, and field of view
B = 9  # Baseline in cm
f = 6  # Focal length in mm
alpha = 56.6  # Field of view in degrees

# Frame rate (max is 120 fps)
frame_rate = 120

# Open both cameras with error handling
print("Checking camera availability...")

def open_camera(index, backend):
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        print(f"Error: Unable to access camera {index} with backend {backend}.")
        return None
    print(f"Camera {index} opened successfully.")
    return cap

# Try different backends (CAP_DSHOW, CAP_MSMF, etc.)
cap_right = open_camera(1, cv2.CAP_DSHOW) or open_camera(1, cv2.CAP_MSMF)
cap_left = open_camera(0, cv2.CAP_DSHOW) or open_camera(0, cv2.CAP_MSMF)

if not cap_right or not cap_left:
    print("Exiting: Cameras could not be initialized.")
    sys.exit()

# Main loop
print("Starting video capture...")
count = -1

while True:
    count += 1
    ret_right, frame_right = cap_right.read()
    ret_left, frame_left = cap_left.read()

    if not ret_right or not ret_left:
        print("Error: Unable to capture frames.")
        break

    # Applying HSV filter
    mask_right = hsv.add_HSV_filter(frame_right, 1)
    mask_left = hsv.add_HSV_filter(frame_left, 0)

    # Show filtered masks for debugging
    cv2.imshow("Filtered Right", mask_right)
    cv2.imshow("Filtered Left", mask_left)

    res_right = cv2.bitwise_and(frame_right, frame_right, mask=mask_right)
    res_left = cv2.bitwise_and(frame_left, frame_left, mask=mask_left)

    # Applying shape recognition
    circles_right = shape.find_circles(frame_right, mask_right)
    circles_left = shape.find_circles(frame_left, mask_left)

    # Print circles for debugging
    print(f"Circles Right: {circles_right}")
    print(f"Circles Left: {circles_left}")

    if circles_right is None or circles_left is None:
        cv2.putText(frame_right, "TRACKING LOST", (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame_left, "TRACKING LOST", (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        # Calculate depth using triangulation
        depth = tri.find_depth(circles_right, circles_left, frame_right, frame_left, B, f, alpha)

        cv2.putText(frame_right, "TRACKING", (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124, 252, 0), 2)
        cv2.putText(frame_left, "TRACKING", (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124, 252, 0), 2)
        cv2.putText(frame_right, f"Distance: {round(depth, 3)} cm", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124, 252, 0), 2)
        cv2.putText(frame_left, f"Distance: {round(depth, 3)} cm", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124, 252, 0), 2)
        print(f"Depth: {depth}")

    # Show the frames for debugging
    cv2.imshow("Frame Right", frame_right)
    cv2.imshow("Frame Left", frame_left)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# Release resources and close windows
print("Releasing resources...")
cap_right.release()
cap_left.release()
cv2.destroyAllWindows()
print("Program terminated.")