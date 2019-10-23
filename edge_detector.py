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
	
def canny(img, lo_thresh=40, hi_thresh=220, sobel_size=3):
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
