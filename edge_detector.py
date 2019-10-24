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


def canny_vanilla(img, lo_thresh=40, hi_thresh=220, sobel_size=3):
    """
    TODO
    """
    return cv2.Canny(img, lo_thresh, hi_thresh, apertureSize=sobel_size,
                     L2gradient=True)


def canny_otsu(img, sobel_size=3):
    ret, hi_threshold = cv2.threshold(img, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    flo_hi_ratio = 0.3
    return cv2.Canny(img, flo_hi_ratio * hi_threshold, hi_threshold,
                     apertureSize=sobel_size, L2gradient=False)


def canny_gaussian_blur(img, lo_thresh=0, hi_thresh=0, sobel_size=3):
    i_gaus_kernel_size = 11
    img_filt = cv2.GaussianBlur(img, (i_gaus_kernel_size, i_gaus_kernel_size),
                                0)

    i_reduc_factor = 2
    i_start = i_reduc_factor // 2
    img_reduc = img_filt[i_start::i_reduc_factor, i_start::i_reduc_factor]

    # If no threshold specified, use the computed median
    if lo_thresh == 0 and hi_thresh == 0:
        # compute the median of the single channel pixel intensities
        med = np.median(img_reduc)
        # apply automatic Canny edge detection using the computed median
        sigma = 0.3
        lo_thresh = int(max(0, (1.0 - sigma) * med))
        hi_thresh = int(min(255, (1.0 + sigma) * med))
    return cv2.Canny(img_reduc, lo_thresh, hi_thresh, apertureSize=sobel_size,
                     L2gradient=True)
