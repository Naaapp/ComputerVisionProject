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
    
    e = cv2.erode(img, kernel, borderType = cv2.BORDER_CONSTANT, iterations = 1) 
    d = cv2.dilate(img, kernel, borderType = cv2.BORDER_CONSTANT, iterations = 1)
    #return d-e # This is not the Beucher Gradient but the non-linear Laplacian
    return cv2.bitwise_not(d+e) # ref : https://stackoverflow.com/questions/19580102/inverting-image-in-python-with-opencv

def joachimLaplacianLinear(img, k1=5, k2=5):
	# TODO remove this function after Quentin has done it.
    kernel = np.ones((k1, k2), np.uint8)
    
    e = cv2.erode(img, kernel, borderType = cv2.BORDER_CONSTANT, iterations = 1) 
    d = cv2.dilate(img, kernel, borderType = cv2.BORDER_CONSTANT, iterations = 1)
    return d-e
	

def canny_vanilla(img, lo_thresh=40, hi_thresh=220, sobel_size=3):
    """
    TODO
    """
    return cv2.Canny(img, lo_thresh, hi_thresh, apertureSize=sobel_size,
                     L2gradient=True)
    
def nonLinearLaplacian(img, kernel_type=cv2.MORPH_RECT, k1=5, k2=5):
    """
    """
    kernel = cv2.getStructuringElement(kernel_type, (k1, k2))
    return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    
def sobel(img, dx=1, dy=1, kernel_size=3):
    """
    """
    return cv2.Sobel(img, cv2.CV_8U, dx, dy, kernel_size)
    
# Tests
if __name__ == "__main__":
    img = cv2.imread("image_database/Building.png", cv2.IMREAD_GRAYSCALE)
    
    cv2.imshow("Original", img)
    cv2.imshow("Beucher", gradientOfBeucher(img))
    cv2.imshow("Canny", canny(img))
    cv2.imshow("NL_Lap", nonLinearLaplacian(img))
    cv2.imshow("Sobel", sobel(img))

    cv2.waitKey(0);
    cv2.destroyAllWindows()


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
