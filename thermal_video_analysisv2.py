# Change video input and video output paths accordingly.
# Image Blurring, Eroding, Dilating, Contour detection and bounding box printing have tunable parameters.
# Uncomment cv2.imshow comments for each image operation to view the result - if required
# Uncomment cv2.VideoWriter object and associated commands if saving video with detections
# Uncomment processing time calculation lines - if required

#   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *  #

import cv2
import numpy as np
import time

# Input path to video
video_path = "path_to_your_video"

# Path for output video
#output_path = "path_to_output"

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
    img_blurred = cv2.GaussianBlur(img_gray, (7,7), 125, 275) # tune kernel size and thresholds for better results
    #cv2.imshow('Blurred Image', img_blurred)

    # Simple Thresholding (Pixel Intensity)
    threshold, simple_thresh = cv2.threshold(img_blurred, threshold, 255, cv2.THRESH_BINARY)
    #cv2.imshow('Simple Thresholded Image', simple_thresh)

    # Erode the image
    img_eroded = cv2.erode(simple_thresh, (9,9), iterations=12) # tune kernel size and number of iterations
    #cv2.imshow('Eroded Image', img_eroded)

    # Dilation of the image
    img_dilated = cv2.dilate(img_eroded, (5,5), iterations=5) # same here
    #cv2.imshow('Dilated Image', img_dilated)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(img_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialise list of contour areas
    contour_areas = []

    # Draw bounding boxes around the contours
    for contour in contours:
        # Registering contour areas to filter by size for coordinate extraction
        if i%10 == 0:
            area = cv2.contourArea(contour)
            contour_areas.append(area)

        # Filter contours by number of points - associated with size and complexity
        if len(contour)<30: # tunable parameter
            continue

        # Get area of contour to filter for size
        area = cv2.contourArea(contour)

        # Filter contours for area
        if area<100: # tunable parameter
            continue

        # Get the bounding rectangle for each contour
        x, y, w, h = cv2.boundingRect(contour)
        # Draw the bounding rectangle on the copy of the original image
        img_boxes = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show frame with bounding boxes
    cv2.imshow('Hot Spots', frame)

    # Extracting largest contour and its centroid for every 10th frame
    if i%10 == 0:
        if contour_areas:
            largest_contour = max(contour_areas)
            idx = contour_areas.index(largest_contour)
            print("Largest contour in this frame =", largest_contour, "\n contour index:", idx)

            x_max, y_max, w_max, h_max = cv2.boundingRect(contours[idx])
            centroid = calculateCentroid(x_max, y_max, x_max + w_max, y_max + h_max)
            #print(centroid)

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
