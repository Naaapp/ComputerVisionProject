import cv2
import numpy as np
import matplotlib.pyplot as plt
import edge_detector as ed
import segment_detector as sd

def cut_hsv(img, h_min=0, h_max=179, s_min=0, s_max=255, v_min=0, v_max=255):
    """
    Filters the image. Keeps only hsv values that are between two thresholds. 
    
    :param img: [np.array] The input image.
    :h_min:     [int] Hue min threshold
    :h_max:     [int] Hue max threshold
    :s_min:     [int] Saturation min threshold
    :s_max:     [int] Saturation max threshold
    :v_min:     [int] Value min threshold
    :v_max:     [int] Value max threshold

    :return:            [np.array] the filtered image in greyscale
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    low = np.array([h_min, s_min, v_min])
    upp = np.array([h_max, s_max, v_max])
    
    mask = cv2.inRange(hsv, low, upp)
    img_mask = cv2.bitwise_and(img, img, mask=mask)
    
    # returns the value channel = greyscale
    return img_mask[:, :, 2]
    
if __name__ == "__main__":
    img = cv2.imread("image_database/soccer.png")
    
    """
        COLOR SELECTION in HSV color space
        
        First method:
            Hypothesis: The lines are white. Thus, for all values of hue, we
                        take small saturation values and high value (light level)
                        values
            Results: A lot of points are still visible in the public and the
                     terrain lines are cut in small chunks.
       
        Second method:
            Hypothesis: The lines are white and on a green surface. The ideas
                        are the same as for the first method but the hue is taken
                        on green values only.
            Results: The public and the score almost disapear. The terrain lines
                     are sharp but the goal is less visible.
        
        Third method:
            Hypothesis : The lines are white on a green surface and enlightened
                         by mostly blue light (outdoor)
            Results: The goal is now visible but a lot of details are now also
                     visible.
        
        Choice : Second method
    """
    
    fig_color = plt.figure(figsize=(12, 8))
    fig_color.add_subplot(2, 2, 1)
    plt.imshow(img)
    plt.title("Original image")
    
    # First method
    imall = cut_hsv(img, 0, 179, 0, 50, 180, 255)
    fig_color.add_subplot(2, 2, 2)
    plt.imshow(imall, cmap='gray', vmin=0, vmax=255)
    plt.title("First method (all hues)\n[0, 0, 180] - [179, 50, 255]")
    
    # Second method
    imgreen = cut_hsv(img, 30, 90, s_max=70, v_min=150)
    fig_color.add_subplot(2, 2, 3)
    plt.imshow(imgreen, cmap='gray', vmin=0, vmax=255)
    plt.title("First method (hue = green)\n[30, 0, 150] - [90, 70, 255]")
    
    # Third method
    imegreen = cut_hsv(img, 30, 130, s_max=70, v_min=150)
    fig_color.add_subplot(2, 2, 4)
    plt.imshow(imegreen, cmap='gray', vmin=0, vmax=255)
    plt.title("First method (hue = green + blue)\n[30, 0, 150] - [130, 70, 255]")    
    
    plt.show()
    
    """
        EDGE DETECTION on second method
        It seems to work way better on the prefiltered image
    """
    
    fig_edge = plt.figure(figsize=(8, 8))
    
    # Canny on the original image
    canny = ed.canny_gaussian_blur(img)
    fig_edge.add_subplot(1, 2, 1)
    plt.imshow(canny, cmap='gray', vmin=0, vmax=255)
    plt.title("Canny on original image")
    
    # Canny on the green image (second method)
    cannygreen = ed.canny_gaussian_blur(imgreen)
    fig_edge.add_subplot(1, 2, 2)
    plt.imshow(cannygreen, cmap='gray', vmin=0, vmax=255)
    plt.title("Canny on pre-filtered image")
    
    plt.show()
    
    """
        SEGMENT DETECTION
        TODO: choisir correctement les paramètres de la hough
    """
    _, res, _ = sd.hough(cannygreen, 1, np.pi/180.0, thresh=50, minLineLen=5, maxLineGap=0, 
                         fuse=True, dTheta=2.0/360.0*np.pi*2.0, dRho=2)
    fig_segment = plt.figure(figsize=(8, 8))
    plt.imshow(res)
    plt.title("Hough on pre-filtered image\nTODO: parameters fiting")
    
    plt.show()   
