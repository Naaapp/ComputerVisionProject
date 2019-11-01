# =====================================================================================================================
# Edge detector for the first part
# =====================================================================================================================

import cv2
import numpy as np


def gradientOfBeucher(img, k1=5, k2=5):
    """
    TODO
    """
    kernel = np.ones((k1, k2), np.uint8)

    e = cv2.erode(img, kernel, borderType=cv2.BORDER_CONSTANT, iterations=1)
    d = cv2.dilate(img, kernel, borderType=cv2.BORDER_CONSTANT, iterations=1)
    return d + e


def canny_vanilla(img, lo_thresh=40, hi_thresh=220, sobel_size=3):
    """
    Apply the canny method to the image (without any preprocessing)
    :param img:         [np.array] The  input image.
    :param lo_thresh:   [int] Low Threshold :  Any edges with intensity
                        gradient lower than this value are sure to be non-edges
    :param hi_thresh:   [int] High Threshold : Any edges with intensity
                        gradient more than this value are sure to be edges
    :param sobel_size:  [int] Size of the Sobel kernel used
                        to get first derivative
    :return:            [np.array] the image containing the local edge points
    """
    return cv2.Canny(img, lo_thresh, hi_thresh, apertureSize=sobel_size,
                     L2gradient=True)


def canny_gaussian_blur(img, lo_thresh=0, hi_thresh=0, sobel_size=3):
    """
    Apply the canny method to the image (with gaussian blur pre-processing)
    :param img:         [np.array] The input image.
    :param lo_thresh:   [int] Low Threshold :  Any edges with intensity
                        gradient lower than this value are sure to be non-edges
    :param hi_thresh:   [int] High Threshold : Any edges with intensity
                        gradient more than this value are sure to be edges
    :param sobel_size:  [int] Size of the Sobel kernel used
                        to get first derivative
    :return:            [np.array] the image containing the local edge points
    """
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


def canny_median_blur(img, lo_thresh=0, hi_thresh=0, sobel_size=3):
    """
    Apply the canny method to the image (with median blur pre-processing)
    :param img:         [np.array] The input image.
    :param lo_thresh:   [int] Low Threshold :  Any edges with intensity
                        gradient lower than this value are sure to be non-edges
    :param hi_thresh:   [int] High Threshold : Any edges with intensity
                        gradient more than this value are sure to be edges
    :param sobel_size:  [int] Size of the Sobel kernel used
                        to get first derivative
    :return:            [np.array] the image containing the local edge points
    """
    i_gaus_kernel_size = 5
    img_filt = cv2.medianBlur(img, i_gaus_kernel_size)

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


def nonLinearLaplacian(img, kernel_type=cv2.MORPH_RECT, k1=5, k2=5):
    """
    Apply the non linear Laplacian to an image.
    """
    kernel = cv2.getStructuringElement(kernel_type, (k1, k2))
    return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

def edgesNLL(img):
	"""
	Take a gray image and return a gray image of the edges detected using a tuned non Linear Laplacian.
	:param img:         [np.array of shape (? x ?)] The  input image.
	:return:            [np.array] the image containing the local edge points
	"""
	GRADIENT_K_SIZE = 2
	BLUR_K_SIZE = 7
	BLUR_SIGMA = 2
	img_float = img.astype(np.uint8)

	# apply gaussian blur to remove noise
	imgBlur = cv2.GaussianBlur(img_float, (BLUR_K_SIZE, BLUR_K_SIZE), BLUR_SIGMA)

	# detect edges
	imgEdges = nonLinearLaplacian(imgBlur, kernel_type=cv2.MORPH_RECT, k1=GRADIENT_K_SIZE, k2=GRADIENT_K_SIZE)

	# apply threshold
	med = np.mean(imgEdges)
	lo_thresh = int(2.5 * med)
	threshValue, imgThresh = cv2.threshold(imgEdges,lo_thresh,255,cv2.THRESH_BINARY)

	return imgThresh


def sobel(img, dx=1, dy=1, kernel_size=3):
    """
    """
    return cv2.Sobel(img, cv2.CV_8U, dx, dy, kernel_size)


# Tests
if __name__ == "__main__":
    img = cv2.imread("image_database/Road.png", cv2.IMREAD_GRAYSCALE)

    # cv2.imshow("Original", img)
    # cv2.imshow("Beucher", gradientOfBeucher(img))
    cv2.imshow("Canny", canny_gaussian_blur(img))
    cv2.imshow("Canny2", canny_median_blur(img))
    # cv2.imshow("NL_Lap", nonLinearLaplacian(img))
    # cv2.imshow("Sobel", sobel(img))

    cv2.waitKey(0)
    cv2.destroyAllWindows()
