# =====================================================================================================================
# Basic tools for image processing
# =====================================================================================================================

import cv2

# Note: I (Joachim) did an object as we don't really know yet how we will store the image. Using our own class allows to
# easily make modification in the code by simply modifying this object instead of each line of code using the image.
class ImgObj:
	""" Object containing the image. """
	
	def __init__(self):
		""" Init."""
		self.path = None 
		self.loaded = False
		self.im = None
	
	def __init__(self, path):
		""" Init with path and load image from it. """
		self.path = path
		self.loaded = False
		self.im = None
		self.load()
		
	def load(self, path=None):
		""" Load the image from the path given or from the last path used/provided.	This will delete the previously 
		loaded image. """
		
		if path is None and self.path is None:
			print("The object has no path to load the image from.")
		else:
			self.img = cv2.imread(self.path) if path is None else cv2.imread(path)
			if (not path is None) : self.path = path
			self.loaded = True
	
	def display(self, name=''):
		""" Display the image loaded in this object. """
		if not self.loaded:
			if self.path is None:
				print("The object has no image loaded and no path to load from.")
			else:
				self.load()
				cv2.imshow(name, self.img)
				print("image displayed")
				cv2.waitKey(0)
				cv2.destroyAllWindows()
				
		else:
			cv2.imshow(name, self.img)
			print("image displayed")
			cv2.waitKey(0)
			cv2.destroyAllWindows()
