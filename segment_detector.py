# =====================================================================================================================
# Segment and endpoints detector for the first part
# =====================================================================================================================

import cv2
import numpy as np

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
	imgEdges = fctEdges(img)
	
	# First stage of voting process
	
	
	# TODO change this to the true value
	seg = imgEdges.copy()
	endPoint = imgEdges.copy()
	
	return seg, endPoint
