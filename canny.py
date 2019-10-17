import cv2
import numpy as np
from matplotlib import pyplot as plt
from ipywidgets import interact, fixed
import matplotlib.cm as cm
import tools

img = cv2.imread('image_database/Road.png', cv2.IMREAD_GRAYSCALE)
lo_thresh = 40
hi_thresh = 220
sobel_size = 3
img_canny = cv2.Canny(img, lo_thresh, hi_thresh, apertureSize=sobel_size,
                      L2gradient=True)
tools.multiPlot(1, 2, (img, img_canny),
                ('Original image', 'Canny ' + str(lo_thresh)),
                cmap_tuple=(cm.gray, cm.gray))
