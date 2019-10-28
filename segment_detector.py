# =====================================================================================================================
# Segment and endpoints detector for the first part
# =====================================================================================================================

import cv2
import numpy as np
import edge_detector as ed


def SegHoughVariant(img, fctEdges, rho=1, theta=np.pi / 180, thresh=50, minLineLen=0, maxLineGap=0, kSize=2):
    """
    Apply the segment detection by preprocessing the image with the edge detection and using the Hough Variant.

    @Args:
        img:		[np.array]
        fctEdges:	[python function] Function taking the img as argument and returning the edge detection of the image.
                    The edges are of value 255 and the rest is at 0.
        rho:		[double] resolution of the image
        theta: 		[double] The resolution of the parameter in radians. We use 1 degree
        thresh:  [int] The minimum number of intersections to “detect” a line
        minLineLen: [double] The minimum number of points that can form a line. Lines with less than this number of
                    points are disregarded.
        maxLineGap: [double] The maximum gap between two points to be considered in the same line.
        kSize:		[int] Size of kernel for dilation

    @Return:
        seg:		[np.array] the image of the segment detected
        endPoint:	[np.array] the image of the endpoints detected
    """
    # Detect the edges
    img_edges = fctEdges(img)
    
    # Dilate edges
    kernel = np.ones((kSize, kSize), np.uint8)
    img_edges = cv2.dilate(img_edges, kernel, borderType=cv2.BORDER_CONSTANT, iterations=1)
    
    # Detect segments of lines
    img_lines_p, img_lines_only = HoughVariant(img_edges, rho, theta, thresh, minLineLen, maxLineGap)
    
    #img_lines_p = cv2.dilate(img_lines_p, kernel, borderType=cv2.BORDER_CONSTANT, iterations=1)
    #img_lines_only = cv2.dilate(img_lines_only, kernel, borderType=cv2.BORDER_CONSTANT, iterations=1)
    
	        
    return img_lines_p, img_lines_only

def HoughVariant(img, rho=1, theta=np.pi / 180, thresh=50, minLineLen=0, maxLineGap=0):
    """
    Apply the Hough Variant on the image.

    @Args:
        img:		[np.array]
        rho:		[double] resolution of the image
        theta: 		[double] The resolution of the parameter in radians. We use 1 degree
        thresh:  [int] The minimum number of intersections to “detect” a line
        minLineLen: [double] The minimum number of points that can form a line. Lines with less than this number of
                    points are disregarded.
        maxLineGap: [double] The maximum gap between two points to be considered in the same line.

    @Return:
        seg:		[np.array] the image of the segment detected
        endPoint:	[np.array] the image of the endpoints detected
    """
    # Copy edges to the images that will display the results in BGR
    img_lines_p = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_lines_only = img_lines_p*0
	
	# Detect segment of lines
    lines_p = cv2.HoughLinesP(img, rho = rho, theta=theta, threshold=thresh, minLineLength=minLineLen, 
    						  maxLineGap=maxLineGap)
	
	# Add segment detected to images
    if lines_p is not None:
        for i in range(0, len(lines_p)):
            line = lines_p[i][0]
            cv2.line(img_lines_p, (line[0], line[1]), (line[2], line[3]),
                     (0, 0, 255), 1)
            cv2.line(img_lines_only, (line[0], line[1]), (line[2], line[3]),
                     (0, 0, 255), 1)        
    return img_lines_p, img_lines_only


def edgesDetectionFinal(img):
    imgEdges = ed.canny_median_blur(img)
    return imgEdges  # imgThresh


if __name__ == "__main__":
    img = cv2.imread("tutorial/Images/boat.png", cv2.IMREAD_GRAYSCALE)

    seg, lines = HoughVariant(img, edgesDetectionFinal)

    cv2.imshow("Segment detection - Variant of Hough transform", seg)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
