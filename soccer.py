import cv2
import numpy as np
import edge_detector as ed
import segment_detector as sd

def go(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    low = np.array([0, 0, 180])
    upp = np.array([179, 50, 255])
    
    mask = cv2.inRange(hsv, low, upp)
    img_mask = cv2.bitwise_and(img, img, mask=mask)

    return img_mask[:, :, 2]
    
if __name__ == "__main__":
    img = cv2.imread("image_database/soccer.png")
    dsp = cv2.pyrDown(img)
    gau = cv2.GaussianBlur(img, (3, 3), 2)
    
    can = ed.canny_vanilla(img)
    non = ed.canny_vanilla(go(img))
    
    _, res, _ = sd.hough(non, 1, np.pi/180.0, thresh=50, minLineLen=5, maxLineGap=0, 
                         fuse=True, dTheta=2.0/360.0*np.pi*2.0, dRho=2)
    
    cv2.imshow("none", non)
    cv2.imshow("dsp", go(dsp))
    cv2.imshow("gau", go(gau))
    cv2.imshow("canny", can)
    cv2.imshow("res", res)
    cv2.waitKey()
    cv2.destroyAllWindows()
