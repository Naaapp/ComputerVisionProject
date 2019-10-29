import numpy as np                     # Numerical algorithms on arrays
import cv2                             # OpenCV
from matplotlib import pyplot as plt   # Plot library
from pylsd.lsd import lsd              # LSD.py python binding

def lsd_alg(gray_image, line_width=0):
    """TODO"""

    result = np.copy(gray_image)
    lines = lsd(gray_image)  # python script calling the C++ so library

    for i in range(lines.shape[0]):
        pt1 = (int(lines[i, 0]), int(lines[i, 1]))
        pt2 = (int(lines[i, 2]), int(lines[i, 3]))
        if line_width == 0:
            width = lines[i, 4]
        else:
            width = line_width * 2
        cv2.line(result[0], pt1, pt2, (0, 0, 255), int(np.ceil(width / 2)))  # CAUTION THIS FUNCTION FAILED NEED TO BE PATCHED (je suis trop fatigué mtn)

    return result[0]  # Lines over a Black background
