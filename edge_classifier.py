# =====================================================================================================================
# Edge point classification for the first part
# =====================================================================================================================

import cv2
import numpy as np
import segment_detector as sd


def edge_classifier(input_img):
    """
    Take an image, detect the edge and segments and classify edges between
    belonging to a segment or not
    :param      input_img:  [np.array] The input image
    :return:    segments:   [np.array] the image of the segment detected with
                            the edges detected previously
                img_edges:  [np.array] the image with the edges
                img_edges_lines:    [np.array] the image with only
                                    the edges belonging to a segment
                img_edges_not_lines:[np.array] the image with only the edges
                                    not belonging to a segment
    """
    img_edges, _, segments, segments_only = sd.segmentDetectorFinal(input_img)
    img_edges_lines = np.multiply(img_edges, segments_only)*255
    img_edges_not_lines = img_edges - img_edges_lines
    return segments, img_edges, img_edges_lines, img_edges_not_lines


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
