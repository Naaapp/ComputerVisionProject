# =====================================================================================================================
# Edge point classification for the first part
# =====================================================================================================================

import cv2
import numpy as np
import edge_detector as ed
import segment_detector as sd
from numpy import arccos, array, dot, pi, cross
from numpy.linalg import det, norm
import math


# from: https://gist.github.com/nim65s/5e9902cd67f094ce65b0
def distance_numpy(A, B, P):
    """ segment line AB, point P, where each one is an array([x, y]) """
    if all(A == P) or all(B == P):
        return 0
    if arccos(dot((P - A) / norm(P - A), (B - A) / norm(B - A))) > pi / 2:
        return norm(P - A)
    if arccos(dot((P - B) / norm(P - B), (A - B) / norm(A - B))) > pi / 2:
        return norm(P - B)
    return norm(cross(A - B, A - P)) / norm(B - A)


def edge_classifier(input_img):
    img_edges = sd.edgesDetectionFinal(input_img)
    seg, lines = sd.HoughVariant(img, sd.edgesDetectionFinal)
    img_edges_lines = np.zeros(shape=np.shape(img_edges), dtype=np.uint8)
    img_edges_not_lines = np.zeros(shape=np.shape(img_edges), dtype=np.uint8)

    h = img_edges.shape[0]
    w = img_edges.shape[1]
    for y in range(0, h):
        for x in range(0, w):
            if img_edges[x][y] == 255:
                point = np.array([x, y])
                min_distance = math.inf
                for line in lines:
                    # print(line)
                    min_distance = min(min_distance, distance_numpy(np.array(
                        [line[0][0], line[0][1]]),
                        np.array([line[0][2], line[0][3]]), point))
                if min_distance < 5:
                    img_edges_lines[x][y] = img_edges[x][y]
                else:
                    img_edges_not_lines[x][y] = img_edges[x][y]
    return seg, img_edges, img_edges_lines, img_edges_not_lines


if __name__ == "__main__":
    img = cv2.imread("tutorial/Images/boat.png", cv2.IMREAD_GRAYSCALE)
    seg, edges, line_edges, not_line_edges = edge_classifier(img)

    cv2.imshow("Segment classification : segment detected by the segment "
               "detector ", seg)
    cv2.imshow("Segment classification : edges ", edges)
    cv2.imshow("Segment classification : edges from line", line_edges)
    cv2.imshow("Segment classification : edges not from line",
               not_line_edges)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
