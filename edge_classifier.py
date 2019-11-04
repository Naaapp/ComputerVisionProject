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
    :param      input_img:          [np.array] The input image
    :return:    img_edges_segment:  [np.array] the image of the segment
                                    detected with the edges detected previously
                img_edges:          [np.array] the image with the edges
                img_edges_lines:    [np.array] the image with only
                                    the edges belonging to a segment
                img_edges_not_lines:[np.array] the image with only the edges
                                    not belonging to a segment
    """
    img_edges, _, img_edges_segment, img_segment = sd.segmentDetectorFinal(
                                                    input_img)
    img_line_edges = np.multiply(img_edges, img_segment)*255
    img_not_line_edges = img_edges - img_line_edges
    return img_edges_segment, img_edges, img_line_edges, img_not_line_edges
