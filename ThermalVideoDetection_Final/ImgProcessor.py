# "/////" - indicates parts of the code that have been commented out because they have no utility in the present scenario but may
# have some in the future

# "&&&&&" - indicates parts of the code that can be uncommented to show the image processing steps. This is useful for debugging,
# improving or just observing the performance of the code i.e., detection efficacy, processing times

# "%%%%%" - indicates parts of the code that need to be uncommented to save the results to a video file

import cv2
import numpy as np
from util import calculateContourCentroid, calculateLinearCentroid


class ImageProcessor:
    def __init__(self, G_Kernel, E_Kernel, D_Kernel, E_Iter, D_Iter, Cont_Size, Cont_Cmplx):
        self.G_Kernel = G_Kernel
        self.E_Kernel = E_Kernel
        self.D_Kernel = D_Kernel
        self.E_Iter = E_Iter
        self.D_Iter = D_Iter
        self.Cont_Size = Cont_Size
        self.Cont_Cmplx = Cont_Cmplx

    def FireDetect(self, frame, i=None):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate mean and standard deviation
        mean = np.mean(frame_gray)
        std_dev = np.std(frame_gray)

        # Establish threshold
        threshold = int(mean+(std_dev*2))

        # Set failsafe for threshold
        if threshold>255:
            threshold = int(mean+std_dev)
        elif threshold<125:
            threshold = 125

        frame_blur = cv2.GaussianBlur(frame_gray, self.G_Kernel, 0)

        # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        # cv2.imshow('Blurred Frame', frame_blur)
        # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        # Simple Thresholding (Pixel Intensity)
        threshold, frame_thresh = cv2.threshold(frame_blur, threshold, 255, cv2.THRESH_BINARY)

        # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        # cv2.imshow('Simple Thresholded Image', simple_thresh)
        # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        # Erode the image
        frame_eroded = cv2.erode(frame_thresh, self.E_Kernel, self.E_Iter)

        # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        # cv2.imshow('Eroded Image', img_eroded)
        # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        # Dilation of the image
        frame_dilated = cv2.dilate(frame_eroded, self.D_Kernel, self.D_Iter)

        # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        # cv2.imshow('Dilated Image', img_dilated)
        # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(frame_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # ///////////////////////////////////////////////////////////////////
        # Initialise list of centroids for every contour and centroid of main contour
        # centroids = []
        # main_contour_points = []
        # main_contour_centroid = []
        # ///////////////////////////////////////////////////////////////////

        # Draw bounding boxes around the contours
        for contour in contours:
            # Filter contours by number of points - associated with size and complexity
            if len(contour) < self.Cont_Cmplx:
                continue

            # Get area of contour to filter for size
            area = cv2.contourArea(contour)

            # Filter contours for area
            if area < self.Cont_Size:
                continue
        
            # ///////////////////////////////////////////////////////////////////
            # Calculate centroids of each contour
            # if i%(frame_step) == 0: # tunable parameter
            #     (cX, cY) = calculateContourCentroid(contour)
            #     centroids.append((cX, cY))
            # ///////////////////////////////////////////////////////////////////
        
            # Get the bounding rectangle for each contour
            x, y, w, h = cv2.boundingRect(contour)
            # Draw the bounding rectangle on the copy of the original image
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # ///////////////////////////////////////////////////////////////////
            # if len(centroids)>=1:
            #     if len(centroids)>=3:
            #         main_contour_points = np.array(centroids)
            #         cv2.drawContours(frame, [main_contour_points], -1, (255, 0, 0), 2)
            #         main_contour_centroid = calculateContourCentroid(main_contour_points)
        
            #     elif len(centroids)==2:
            #         main_contour_centroid = calculateLinearCentroid(centroids)
        
        #     elif len(centroids)==1:
        #         main_contour_centroid = centroids[0]

        # Show Area of Interest centroid
        # if main_contour_centroid:
        #     if all(point>0 for point in main_contour_centroid):
        #         print(main_contour_centroid)

        # Increment frame extraction counter
        # i +=1
        # ///////////////////////////////////////////////////////////////////
    
        return frame