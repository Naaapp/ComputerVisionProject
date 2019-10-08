# =====================================================================================================================
# Main program for the Computer Vision project.
# =====================================================================================================================

import argparse
import image_tools as it
import image_processor as ip


if __name__ == "__main__":
	string = '''Arguments to launch the different parts of the assignment.'''
	parser = argparse.ArgumentParser(description=string)
	
	parser.add_argument("-ip", "--im_path", type=str,
						help="path to the current image.", default=None, )#required=True) # Should it be a required 
																					      # arg or not?

	parser.add_argument("-1.1", "--display", action="store_true",
						help="display the current image and the result of the processing.", default=False)
	
	
	
	args = parser.parse_args()
	
	path = args.im_path
	
	# ===========================================
	#                   Part 1
	# ===========================================	
	# 1.1. Display
	# ==================
	if args.display:
		if path is None:
			path = input("Please provide the path to the image to display : ")
		
		img = it.ImgObj(path)
		img.display("Display Test") # TODO change this name to ???
		
		imgPro = ip.edgePntExtr(img)
		imgPro.display("Display after process Test") # TODO change this name to ???
	
