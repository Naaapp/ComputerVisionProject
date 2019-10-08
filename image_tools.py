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
		self.img = None
		self.imgGrey = None
		self.greyExist = False
	
	def __init__(self, path):
		""" Init with path and load image from it. """
		self.path = path
		self.loaded = False
		self.img = None
		self.loadFromPath()
		self.imgGrey = None
		self.greyExist = False
		
	def loadFromPath(self, path=None):
		""" Load the image from the path given or from the last path used/provided.	This will delete the previously 
		loaded image. """
		
		if path is None and self.path is None:
			print("The object has no path to load the image from.")
		else:
			self.img = cv2.imread(self.path) if path is None else cv2.imread(path)
			if (not path is None) : self.path = path
			self.loaded = True
	
	def loadFromImg(self, newImg):
		""" Load the image from the image given. This will delete the previously 
		loaded image. """
		
		self.img = newImg
		self.loaded = True
	
	def display(self, name=''):
		""" Display the image loaded in this object. """
		if not self.loaded:
			if self.path is None:
				print("The object has no image loaded and no path to load from.")
			else:
				self.loadFromPath()
				cv2.imshow(name, self.img)
				cv2.waitKey(0)
				cv2.destroyAllWindows()
				
		else:
			cv2.imshow(name, self.img)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
			
	def toGrey(self):
		if not self.loaded:
			self.loadFromPath()
		self.greyExist = True
		self.imgGrey = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
	
	def getImg(self):
		return self.img
	
	def getGreyImg(self):
		if self.greyExist:
			return self.imgGrey
		else:
			self.toGrey()
			return self.imgGrey
