# =====================================================================================================================
# Processor of the image to detect edges and other.
# =====================================================================================================================

def edgePntExtr(imgObj):
	""" 
	Extract local edges of an images.
	
	@Args:
		imgObj : 	[image_tools.ImgObj] The image from which to extract edges
	@Return:
		[image_tools.ImgObj] The image with the edges extracted ??? TODO
	"""
	#TODO
	imgObjPro = imgObj
	imgObjPro.loadFromImg(imgObj.getGreyImg())
	return imgObjPro
