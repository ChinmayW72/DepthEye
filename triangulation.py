import numpy as np

def find_depth(circle_right, circle_left, frame_right, frame_left, baseline, f, alpha):
    height_right, width_right, _ = frame_right.shape
    height_left, width_left, _ = frame_left.shape

    if width_right != width_left:
        print("Error: Frame widths of cameras do not match.")
        return None

    # Convert focal length to pixels
    f_pixel = (width_right * 0.5) / np.tan(alpha * 0.5 * np.pi / 180)

    x_right = circle_right[0]
    x_left = circle_left[0]

    disparity = x_left - x_right

    if abs(disparity) < 1e-5:  # Prevent division by zero
        print("Error: Disparity too small.")
        return None

    # Calculate depth
    z_depth = (baseline * f_pixel) / disparity
    return abs(z_depth)