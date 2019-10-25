# =====================================================================================================================
# Segment and endpoints detector for the first part
# =====================================================================================================================

import cv2
import numpy as np
import edge_detector as ed


def HoughVariant(img, fctEdges, rho=1, theta=np.pi / 180, thresh=50, minLineLen=0, maxLineGap=0):
    """
    TODO

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

    @Return:
        seg:		[np.array] the image of the segment detected
        endPoint:	[np.array] the image of the endpoints detected
    """
    # Detect the edges
    img_edges = fctEdges(img)

    # Copy edges to the images that will display the results in BGR
    img_lines_p = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2BGR)
    img_lines_only = img_lines_p*0

    lines_p = cv2.HoughLinesP(img_edges, rho = rho, theta=theta, threshold=thresh, minLineLength=minLineLen, 
    						  maxLineGap=maxLineGap)

    if lines_p is not None:
        for i in range(0, len(lines_p)):
            line = lines_p[i][0]
            cv2.line(img_lines_p, (line[0], line[1]), (line[2], line[3]),
                     (0, 0, 255), 1)
            cv2.line(img_lines_only, (line[0], line[1]), (line[2], line[3]),
                     (0, 0, 255), 1)

    return img_lines_p, img_lines_only


def edgesDetectionFinal(img):
    imgEdges = ed.canny_gaussian_blur(img)
    return imgEdges  # imgThresh


if __name__ == "__main__":
    img = cv2.imread("image_database/Building.png", cv2.IMREAD_GRAYSCALE)

    seg, endPoint = HoughVariant(img, edgesDetectionFinal)

    cv2.imshow("Original", endPoint)
    cv2.imshow("Segment detection - Variant of Hough transform", seg)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
