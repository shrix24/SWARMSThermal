import numpy as np
import cv2

def rescaleFrame(frame, scale=0.65):
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)
    # print(width, height)
    dimensions = (width, height)

    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

def calculateContourCentroid(contour):
    moments = cv2.moments(contour)
    cX = int(moments["m10"]/moments["m00"])
    cY = int(moments["m01"]/moments["m00"])
    return (cX, cY)

def calculateLinearCentroid(line):
    cX = line[1][0] - line[0][0]
    cY = line[1][1] - line[0][1]
    return (cX, cY)

