# =====================================================================================================================
# Main program for the Computer Vision project.
# =====================================================================================================================

import argparse
import image_tools


if __name__ == "__main__":
	string = '''Arguments to launch the different parts of the assignment.'''
	parser = argparse.ArgumentParser(description=string)
	
	parser.add_argument("-ip", "--im_path", type=str,
						help="path to the current image.", default=None, )#required=True) # Should it be a required 
																					      # arg or not?

	parser.add_argument("-1.1", "--display", action="store_true",
						help="display the current image and the result of the processing.", default=False)
	
	
	
	args = parser.parse_args()
	
	# ===========================================
	#                   Part 1
	# ===========================================	
	# 1.1. Display
	# ==================
	if args.display:
		path = "image_database/Building.png"
		img = image_tools.ImgObj(path)
		img.display("Test")
	
