import cv2
import numpy as np
import edge_detector as ed

def go(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    low = np.array([0, 0, 180])
    upp = np.array([179, 50, 255])
    
    mask = cv2.inRange(hsv, low, upp)
    img_mask = cv2.bitwise_and(img, img, mask=mask)
    img_grey = img_mask[:, :, 2]
    _, img_thresh = cv2.threshold(img_grey, 40, 255, cv2.THRESH_BINARY)

    return img_thresh
    
if __name__ == "__main__":
    img = cv2.imread("image_database/soccer.png")
    dsp = cv2.pyrDown(img)
    gau = cv2.GaussianBlur(img, (3, 3), 2)
    can = ed.canny_vanilla(img)
    
    cv2.imshow("none", go(img))
    cv2.imshow("dsp", go(dsp))
    cv2.imshow("gau", go(gau))
    cv2.imshow("canny", can)
    cv2.waitKey()
    cv2.destroyAllWindows()
