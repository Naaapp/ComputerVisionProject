# =====================================================================================================================
# Edge detector for the first part
# =====================================================================================================================

import cv2
import numpy as np


def gradientOfBeucher(img, k1=5, k2=5):
    """
    TODO
    """
    kernel = np.ones((k1, k2), np.uint8)  # TODO check other kernels
    return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)


def canny(img, lo_thresh=40, hi_thresh=220, sobel_size=3):
    """
    TODO
    """
    return cv2.Canny(img, lo_thresh, hi_thresh, apertureSize=sobel_size,
                     L2gradient=True)
