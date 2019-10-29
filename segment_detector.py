# =====================================================================================================================
# Segment and endpoints detector for the first part
# =====================================================================================================================

import cv2
import numpy as np
import edge_detector as ed
import random


def segHoughVariant(img, fctEdges, rho=1, theta=np.pi / 180, thresh=50, minLineLen=0, maxLineGap=0, kSize=2):
    """
    Apply the segment detection by preprocessing the image with the edge detection and using the Hough Variant.

    @Args:
        img:		[np.array]
        fctEdges:	[python function] Function taking the img as argument and returning the edge detection of the image.
                    The edges are of value 255 and the rest is at 0.
        rho:		[double] resolution of the image
        theta: 		[double] The resolution of the parameter in radians. We use 1 degree
        thresh:  [int] The minimum number of intersections to “detect” a line
        minLineLen: [double] The minimum number of points that can form a line. Lines with less than this number of
                    points are disregarded.
        maxLineGap: [double] The maximum gap between two points to be considered in the same line.
        kSize:		[int] Size of kernel for dilation

    @Return:
        seg:		[np.array] the image of the segment detected
        endPoint:	[np.array] the image of the endpoints detected
    """
    # Detect the edges
    img_edges = fctEdges(img)
    
    # Dilate edges
    kernel = np.ones((kSize, kSize), np.uint8)
    img_edges = cv2.dilate(img_edges, kernel, borderType=cv2.BORDER_CONSTANT, iterations=1)
    
    # Detect segments of lines
    img_lines_p, img_lines_only = houghVariant(img_edges, rho, theta, thresh, minLineLen, maxLineGap)
    
    #img_lines_p = cv2.dilate(img_lines_p, kernel, borderType=cv2.BORDER_CONSTANT, iterations=1)
    #img_lines_only = cv2.dilate(img_lines_only, kernel, borderType=cv2.BORDER_CONSTANT, iterations=1)
    
	        
    return img_lines_p, img_lines_only

def houghVariant(img, rho=1, theta=np.pi / 180, thresh=50, minLineLen=0, maxLineGap=0):
    """
    Apply the Hough Variant on the image.

    @Args:
        img:		[np.array]
        rho:		[double] resolution of the image
        theta: 		[double] The resolution of the parameter in radians. We use 1 degree
        thresh:  [int] The minimum number of intersections to “detect” a line
        minLineLen: [double] The minimum number of points that can form a line. Lines with less than this number of
                    points are disregarded.
        maxLineGap: [double] The maximum gap between two points to be considered in the same line.

    @Return:
        seg:		[np.array] the image of the segment detected
        endPoint:	[np.array] the image of the endpoints detected
    """
    # Copy edges to the images that will display the results in BGR
    img_lines_p = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_lines_only = img_lines_p*0
	
	# Detect segment of lines
    lines_p = cv2.HoughLinesP(img, rho = rho, theta=theta, threshold=thresh, minLineLength=minLineLen, 
    						  maxLineGap=maxLineGap)
	
    print(type(lines_p))
    print(lines_p.shape)
	# Add segment detected to images
    if lines_p is not None:
        for i in range(0, len(lines_p)):
            line = lines_p[i][0]
            cv2.line(img_lines_p, (line[0], line[1]), (line[2], line[3]),
                     (0, 0, 255), 1)
            cv2.line(img_lines_only, (line[0], line[1]), (line[2], line[3]),
                     (0, 0, 255), 1)        
    return img_lines_p, img_lines_only


def toHoughSpaceVariant(AB):
	"""
	Given the list of the two end points of segments, return the values of the segments in a variant of the Hough space.
	@Args:
		AB:		[numpy array of shape (num seg x 1 x 4)] Array containing the coordinates of the first and second 
				endpoint of segment of line. (Considering the origin in top left and values in order [v1, h1, v2, h2].)
	@Return:
		A list of lists containing :
		theta:	[float] inclination of the slope of the segment in radians
		rho:	[float] shortest distance between the segment (extended to infinity) and the origin. 
		p:		[float] distance from C to the endpoint with the lowest horizontal value. C being the intersection 
				between the segment extended and the perpendicular to the segment going to rho
		d:		[float] distance from A to B
	"""
	retList = []
	
	for i in range(AB.shape[0]):
		endpts = AB[i]
		av = endpts[0][0] # vertical coord. of A
		ah = endpts[0][1] # horizontal coord. of A
		bv = endpts[0][2] # vertical coord. of B
		bh = endpts[0][3] # horizontal coord. of B
		
		d = np.linalg.norm(np.array([bv-av,bh-ah]))
		if bh-ah == 0:
			theta = np.pi/2
		else:
			theta = np.arctan(abs(bv-av)/abs(bh-ah)) # theta in [0, pi/2[
			if ( ah > bh and av > bv) or ( ah < bh and av < bv): # decreasing slope => theta in ]pi/2, pi[
				theta = np.pi - theta
		#print(f'\nTo : {endpts} -')
		rho =  np.linalg.norm(np.cross(np.array([bv-av,bh-ah]), np.array([av, ah]))) / d 
		if av != bv and ah-av*(ah-bh)/(av-bv) < 0:
			rho = -rho # rho < 0 if ch < 0
		cv = rho*np.cos(theta)
		ch = rho*np.sin(theta)
		#print(f'\nTo : {endpts} - C is ({cv}, {ch})')
		if ah < bh or (ah == bh and av > bv):
			p = np.sign(-5+6*np.sign(ah - ch)) * np.linalg.norm(np.array([av-cv,ah-ch])) # p < 0 if ah <= ch
		else:
			p = np.sign(-5+6*np.sign(bh - ch)) * np.linalg.norm(np.array([bv-cv,bh-ch])) # p < 0 if bh <= ch
		
		retList.append([theta, rho, p, d])
		
	return retList

def fromHoughSpaceVariant(abHS):
	"""
	From Hough space variant to the segment endpoints.
	@Return:
		AB:		[numpy array of shape (num seg x 1 x 4)] Array containing the coordinates of the first and second 
				endpoint of segment of line. (Considering the origin in top left and values in order [v1, h1, v2, h2].)
	"""
	retList = np.zeros((len(abHS), 1, 4))
	
	for i in range(len(abHS)):
		pt = abHS[i]
		theta = pt[0]
		rho = pt[1]
		p = pt[2]
		d = pt[3]
		sin = np.sin(theta)
		cos = np.cos(theta)
		
		# finding C, 
		# C being the intersection between the segment extended and the perpendicular to the segment going to rho
		cv = cos*rho # vertical coord of C
		ch = sin*rho # horizontal coord. of C
		#print(f'From  : {pt} - C is ({cv}, {ch})')
		
		# Finding the endpoints A and B
		av = int(round(cv - np.sign(1+2*np.sign(np.pi/2-theta))*p*sin))
		ah = int(round(ch + np.sign(1+2*np.sign(np.pi/2-theta))*p*cos))
		bv = int(round(cv - np.sign(1+2*np.sign(np.pi/2-theta))*(p+d)*sin))
		bh = int(round(ch + np.sign(1+2*np.sign(np.pi/2-theta))*(p+d)*cos))
		
		retList[i][0] = np.array([av,ah, bv, bh])
		
	return retList
	

def fuseCloseSegment(AB):
	"""
	Fuse close segments together.
	@Args:
		AB:		[numpy array of shape (num seg x 1 x 4)] Array containing the coordinates of the first and second 
				endpoint of segment of line. (Considering the origin in top left and values in order [v1, h1, v2, h2].)
	@Return:
		The list of segment with close segments fused together.
	"""
	abHS = toHoughSpaceVariant(AB)
	i = 0
	dTheta = 1/360*np.pi*2 # 1 degre difference of theta accepted
	dRho = 1.5 # 1.5 pixel difference of rho accepted
	
	while True:
		if i == len(abHS):
			break
			
		seg1 = abHS[i]
		removeI = False
		toAdd = []
		toRemove = []
		for j in range(i+1, len(abHS)):
			seg2 = abHS[j]
			
			# Check first condition to fuse
			if abs(seg1[0]-seg2[0]) <= dTheta and abs(seg1[1]-seg2[1]) <= dRho:
				p1 = seg1[2]
				d1 = seg1[3]
				p2 = seg2[2]
				d2 = seg2[3]
				# Check second condition
				if (( p1 <= p2 and p1+d1 > p2 ) or
				    ( p2 <= p1 and p2+d2 > p1 )) :
				
					removeI = True
					
					newTheta = (seg1[0]+seg2[0])/2
					newRho = (seg1[1]+seg2[1])/2
					newP = min(p1, p2)
					newD = max(p1+d1, p2+d2) - newP
					
					toAdd.append([newTheta, newRho, newP, newD])
					toRemove.append(seg2)
					
		# increment new seg to check
		if removeI:
			abHS.remove(seg1)
		else:
			i += 1
		
		# Add new fused segment and remove previous ones
		for seg in toRemove:
			abHS.remove(seg)
		for seg in toAdd:
			abHS.append(seg)
	
	return fromHoughSpaceVariant(abHS)
				
				
	
	

def edgesDetectionFinal(img):
    imgEdges = ed.canny_gaussian_blur(img)
    return imgEdges 

def testing(num):
	toTest = []
	for i in range(num):
		toTest = np.zeros((1,1,4))
		randLim = num/10
		a1 = random.randint(0, randLim)
		a2 = random.randint(0, randLim)
		b1 = random.randint(0, randLim)
		b2 = random.randint(0, randLim)
		while a1 == b1 and a2 == b2:
			b1 = random.randint(0, randLim)
			b2 = random.randint(0, randLim)
			
		toTest[0][0] = np.array([a1,a2, b1,b2])
		if i == 0:
			print(toTest)
			print()
			print()
			
		tested = fromHoughSpaceVariant(toHoughSpaceVariant(toTest))
		ok = True
		
		for j in range(len(toTest)):
			for l in range(4):
				if tested[j][0][l] != toTest[j][0][l] and tested[j][0][l] != toTest[j][0][(l+2)%4]:
					ok=False
					break
		if not ok:
			print(f'Error {i}: \ntoTest = \n{toTest}\n\n tested = \n{tested}')
			break
def UselesshoughVariant(img, fctEdges, rho=1, theta=np.pi / 180, thresh=50, minLineLen=10, maxLineGap=0):
    """
    Apply the Hough Variant on the image.

    @Args:
        img:		[np.array]
        rho:		[double] resolution of the image
        theta: 		[double] The resolution of the parameter in radians. We use 1 degree
        thresh:  [int] The minimum number of intersections to “detect” a line
        minLineLen: [double] The minimum number of points that can form a line. Lines with less than this number of
                    points are disregarded.
        maxLineGap: [double] The maximum gap between two points to be considered in the same line.

    @Return:
        seg:		[np.array] the image of the segment detected
        endPoint:	[np.array] the image of the endpoints detected
    """
    img_edges = fctEdges(img)
    
    # Dilate edges
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.dilate(img_edges, kernel, borderType=cv2.BORDER_CONSTANT, iterations=1)
    
    # Copy edges to the images that will display the results in BGR
    img_lines_p = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_lines_only = img_lines_p*0
	
	# Detect segment of lines
    lines_p = cv2.HoughLinesP(img, rho = rho, theta=theta, threshold=thresh, minLineLength=minLineLen, 
    						  maxLineGap=maxLineGap)
    return lines_p, 0

if __name__ == "__main__":
	img = cv2.imread("image_database/Building.png", cv2.IMREAD_GRAYSCALE)

	seg, endPoint = UselesshoughVariant(img, edgesDetectionFinal)
	toTest = seg
	tested = fromHoughSpaceVariant(toHoughSpaceVariant(seg))
	ok = True
	for j in range(len(toTest)):
		for l in range(4):
			if tested[j][0][l] != toTest[j][0][l] and tested[j][0][l] != toTest[j][0][(l+2)%4]:
				ok=False
				break
		if not ok:
			print(f'Error {i}: \ntoTest = \n{toTest[j]}\n\n tested = \n{tested[j]}')
			break
	print('OK')
	
	print(toTest[45])
	print(tested[45])
	
	toTest = seg
	tested = fuseCloseSegment(seg)
	print(toTest[45])
	print(tested[45])
	print(toTest.shape)
	print(tested.shape)
	
	ok = True
	for j in range(len(toTest)):
		for l in range(4):
			if tested[j][0][l] != toTest[j][0][l] and tested[j][0][l] != toTest[j][0][(l+2)%4]:
				ok=False
				break
		if not ok:
			print(f'Error {j}: \ntoTest = \n{toTest[j]}\n\n tested = \n{tested[j]}')
			break
	print('OK')
	
	print(seg.shape)
	print(fromHoughSpaceVariant(toHoughSpaceVariant(seg)).shape)

	#cv2.imshow("Original", endPoint)
	#cv2.imshow("Segment detection - Variant of Hough transform", seg)

	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	"""
	lines = [
			  [[2,2,4,1]],
			  [[3,1,1,2]],
			  [[3,1,1,2]],
			  [[12,1,10,2]],
			  [[12,3,10,4]],
			  [[12,1,12,5]],
			  [[17,3,10,3]],
			  [[0,7,28,0]],
			  [[4,7,2,0]]
			]
	lines_np = np.array(lines)
	print(lines_np.shape)
	
	for num in [1000, 10000, 20000, 100000]:
		testing(num)
		print(f"Test passed : {num}")
	print()
	
	
	print(lines)
	print(" -------- \n\n")
	print(fromHoughSpaceVariant(toHoughSpaceVariant(lines_np)))
	print(" -------- \n\n")
	print(fromHoughSpaceVariant(toHoughSpaceVariant(np.array([lines_np[-1]]))))
	print(" -------- \n\n")
	#print(fuseCloseSegment(lines))
	print(2.976443976175166*180/np.pi)

	print(np.sin(2.86))
	"""
