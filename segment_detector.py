# =====================================================================================================================
# Segment and endpoints detector for the first part
# =====================================================================================================================

import cv2
import numpy as np
import edge_detector as ed


def HoughVariant(img, fctEdges):
    """
    TODO

    @Args:
        img:		[np.array]
        fctEdges:	[python function] Function taking the img as argument and returning the edge detection of the image.
                    The edges are of value 255 and the rest is at 0.

    @Return:
        seg:		[np.array] the image of the segment detected
        endPoint:	[np.array] the image of the endpoints detected
    """
    # Detect the edges
    img_edges = fctEdges(img)

    # Copy edges to the images that will display the results in BGR
    img_lines_p = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2BGR)

    # First stage of voting process

    lines_p = cv2.HoughLinesP(img_edges, 1, np.pi / 180, 60, None, 10, 10)

    print(len(lines_p))

    if lines_p is not None:
        for i in range(0, len(lines_p)):
            line = lines_p[i][0]
            cv2.line(img_lines_p, (line[0], line[1]), (line[2], line[3]),
                     (0, 0, 255), 1,
                     cv2.LINE_4)

    # # TODO change this to the true value
    # seg = img_edges.copy()
    # endPoint = img_edges.copy()

    return img_lines_p, lines_p


def edgesDetectionFinal(img):
    imgEdges = ed.canny_median_blur(img)
    return imgEdges  # imgThresh


if __name__ == "__main__":
    img = cv2.imread("tutorial/Images/boat.png", cv2.IMREAD_GRAYSCALE)

    seg, lines = HoughVariant(img, edgesDetectionFinal)

    cv2.imshow("Segment detection - Variant of Hough transform", seg)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
