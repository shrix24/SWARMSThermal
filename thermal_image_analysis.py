import cv2
import numpy as np

# Image rescale function with scale parameter
def rescaleFrame(frame, scale=0.65):
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)
    print(width, height)
    dimensions = (width, height)

    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

# Read image
img = cv2.imread("path_to_your image")
cv2.imshow('Original Image', img)
print(img.shape)

# Reshape image if too big
if img.shape[0]>720 or img.shape[1]>1280:
    print("Rescaling image")
    img_scaled = rescaleFrame(img,scale=0.65)
    img = img_scaled

# Show image
#cv2.imshow('Original Image', img)

# Convert the image to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Calculate mean and standard deviation
mean = np.mean(img_gray)
std_dev = np.std(img_gray)
print("Mean pixel intensity =", mean, "\nStd. Deviation=", std_dev)

# Establish threshold
threshold = int(mean+(std_dev*2))

# Set failsafe for threshold
if threshold>255:
    threshold = int(mean+std_dev)

print("Pixel Intensity Threshold =", threshold)

# Perform Gaussian Blurring
img_blurred = cv2.GaussianBlur(img_gray, (7,7), 125, 275)
cv2.imshow('Blurred Image', img_blurred)

# Simple Thresholding (Pixel Intensity)
threshold, simple_thresh = cv2.threshold(img_blurred, threshold, 255, cv2.THRESH_BINARY)
cv2.imshow('Simple Thresholded Image', simple_thresh)

# Erode the image
img_eroded = cv2.erode(simple_thresh, (9,9), iterations=12)
cv2.imshow('Eroded Image', img_eroded)

# Dilation of the image
img_dilated = cv2.dilate(img_eroded, (5,5), iterations=5)
cv2.imshow('Dilated Image', img_dilated)

# Find contours and their hierarchies in the thresholded image
contours, hierarchy = cv2.findContours(img_dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(contours)

# Draw bounding boxes around the contours
for contour in contours:
    if len(contour)<15:
        continue

    area = cv2.contourArea(contour) # Get area of contour to filter for size

    # Filter contours for area
    if area<10:
        continue

    # Get the bounding rectangle for each contour
    x, y, w, h = cv2.boundingRect(contour)
    # Draw the bounding rectangle on the copy of the original image
    img_boxes = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the result with bounding boxes
cv2.imshow('Bounding Boxes around Hot Spots', img_boxes)
cv2.waitKey(0)
cv2.destroyAllWindows()
