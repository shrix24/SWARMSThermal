# Change video input and video output paths accordingly.
# Image Blurring, Eroding, Dilating, Contour detection and bounding box printing have tunable parameters.
# Uncomment cv2.imshow comments for each image operation to view the result - if required
# Uncomment cv2.VideoWriter object and associated commands if saving video with detections
# Uncomment processing time calculation lines - if required

import cv2
import time
import numpy as np

# Declaration of tunable parameters as global variables
global Gaussian_Kernel, Gaussian_thres1, Gaussian_thres2, Erosion_Kernel, Erosion_Iterations, Dilation_Kernel, Dilation_Iterations

# Always set Kernel parameters as tuples/vectors, example: (7,7) and must always be square
Gaussian_Kernel = (15,15)
Gaussian_thres1 = 175
Gaussian_thres2 = 125
Erosion_Kernel = (11,11)
Erosion_Iterations = 12
Dilation_Kernel = (5,5)
Dilation_Iterations = 12

contour_complexity = 30
contour_size = 100
frame_step = 10

def calculateContourCentroid(contour):
    moments = cv2.moments(contour)
    cX = int(moments["m10"]/moments["m00"])
    cY = int(moments["m01"]/moments["m00"])
    return (cX, cY)

def calculateLinearCentroid(line):
    cX = line[1][0] - line[0][0]
    cY = line[1][1] - line[0][1]
    return (cX, cY)

# Input path to video
video_path = "C:\development\Thermal_Fire_Detection\Thermal_images\WhiteHotThermal.mov"

# Path for output video
#output_path = "Fire_Detect_v3.mp4"

# Create Video capture object
video_cap = cv2.VideoCapture(video_path)

# Check if video opened
if not video_cap.isOpened():
    print("Error opening video")
    exit()

# Get video properties
if video_cap.isOpened():
    width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))

# Create video writer object - uncomment to save output video
#fourcc = cv2.VideoWriter.fourcc(*'DIVX')
#output = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Frame processing time collection array - Uncomment to calculate processing time
#processing_time = []

# Counter for frame extraction
i = 0

# Read video frame by frame
while True:
    #start_time = time.time() # uncomment to calculate processing time
    ret, frame = video_cap.read()

    # Exit the loop if there are no frames or if video is empty
    if not ret:
        break

    # Convert the image to grayscale
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate mean and standard deviation
    mean = np.mean(img_gray)
    std_dev = np.std(img_gray)
    #print("Mean pixel intensity =", mean, "\nStd. Deviation=", std_dev)

    # Establish threshold
    threshold = int(mean+(std_dev*2))

    # Set failsafe for threshold
    if threshold>255:
        threshold = int(mean+std_dev)

    # Perform Gaussian Blurring
    img_blurred = cv2.GaussianBlur(img_gray, Gaussian_Kernel, Gaussian_thres1, Gaussian_thres2) # tune kernel size and thresholds for better results
    #cv2.imshow('Blurred Image', img_blurred)

    # Simple Thresholding (Pixel Intensity)
    threshold, simple_thresh = cv2.threshold(img_blurred, threshold, 255, cv2.THRESH_BINARY)
    #cv2.imshow('Simple Thresholded Image', simple_thresh)

    # Erode the image
    img_eroded = cv2.erode(simple_thresh, Erosion_Kernel, Erosion_Iterations) # tune kernel size and number of iterations
    #cv2.imshow('Eroded Image', img_eroded)

    # Dilation of the image
    img_dilated = cv2.dilate(img_eroded, Dilation_Kernel, Dilation_Iterations) # same here
    #cv2.imshow('Dilated Image', img_dilated)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(img_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialise list of centroids for every contour and centroid of main contour
    centroids = []
    main_contour_points = []
    main_contour_centroid = []

    # Draw bounding boxes around the contours
    for contour in contours:
        # Filter contours by number of points - associated with size and complexity
        if len(contour) < contour_complexity: # tunable parameter
            continue

        # Get area of contour to filter for size
        area = cv2.contourArea(contour)

        # Filter contours for area
        if area < contour_size: # tunable parameter
            continue

        # Calculate centroids of each contour
        if i%(frame_step) == 0: # tunable parameter
            (cX, cY) = calculateContourCentroid(contour)
            centroids.append((cX, cY))
        
        # Get the bounding rectangle for each contour
        x, y, w, h = cv2.boundingRect(contour)
        # Draw the bounding rectangle on the copy of the original image
        img_boxes = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if len(centroids)>=1:
        if len(centroids)>=3:
            main_contour_points = np.array(centroids)
            cv2.drawContours(frame, [main_contour_points], -1, (255, 0, 0), 2)
            main_contour_centroid = calculateContourCentroid(main_contour_points)
        
        elif len(centroids)==2:
            main_contour_centroid = calculateLinearCentroid(centroids)
        
        elif len(centroids)==1:
            main_contour_centroid = centroids[0]

    # Show frame with bounding boxes
    cv2.imshow('Hot Spots', frame)

    # Show Area of Interest centroid
    if main_contour_centroid:
        if all(point>0 for point in main_contour_centroid):
            print(main_contour_centroid)

    # Increment frame extraction counter
    i +=1

    # Write frame with bounding boxes to output video - uncomment to save video with detections
    #output.write(frame)

    # Uncomment to calculate processing time for each frame
    #end_time = time.time()
    #processing_time.append(end_time-start_time)

    # Break statement
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Uncomment to print mean frame processing time
#print(np.mean(processing_time))

# Release the video writer object, video capture object and close windows
#output.release() # uncomment if saving output video
video_cap.release()
cv2.destroyAllWindows()
