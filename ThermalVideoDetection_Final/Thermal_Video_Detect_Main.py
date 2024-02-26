# Change video input and video output paths accordingly.
# Image Blurring, Eroding, Dilating, Contour detection and bounding box printing have tunable parameters.
# Uncomment cv2.imshow comments for each image operation to view the result - if required
# Uncomment cv2.VideoWriter object and associated commands if saving video with detections
# Uncomment processing time calculation lines - if required

# "/////" - indicates parts of the code that have been commented out because they have no utility in the present scenario but may
# have some in the future

# "&&&&&" - indicates parts of the code that can be uncommented to show the image processing steps. This is useful for debugging,
# improving or just observing the performance of the code i.e., detection efficacy, processing times

# "%%%%%" - indicates parts of the code that need to be uncommented to save the results to a video file

# Import modules
import cv2
import numpy as np
import time
import subprocess
from ImgProcessor import ImageProcessor
from util import rescaleFrame

# Declaration of tunable parameters as global variables
global G_Kernel, E_Kernel, E_Iter, D_Kernel, D_Iter, Cont_Cmplx, Cont_Size

# Always set Kernel parameters as tuples/vectors, example: (7,7) and must always be square
G_Kernel = (7,7) # Size of the Gaussian Blur Kernel
E_Kernel = (11,11) # Size of the Erosion Kernel
E_Iter = 12 # Number of Erosion Iterations
D_Kernel = (5,5) # Size of the Dilation Kernel
D_Iter = 12 # Number of Dilation Iterations
Cont_Cmplx = 10 # Filter for displaying contours based on complexity
Cont_Size = 100 # Filter for displaying contours based on size

# Input path to video
ffmpeg_cmd = [
    'ffmpeg',
    '-f', 'dshow',           # Use DirectShow as the input format
    '-video_size', '1920x1080',  # Specify the video size
    '-framerate', '30',      # Specify the framerate
    '-vcodec', 'mjpeg',      # Specify the MJPEG codec
    '-i', 'video=USB3. 0 capture',  # Specify the video capture device (replace with your device name)
    '-pix_fmt', 'bgr24',     # Specify the pixel format as BGR24
    '-c:v', 'rawvideo',
    '-an',                   # Disable audio recording
    '-sn',                   # Disable subtitle recording
    '-vcodec', 'rawvideo',
    '-f', 'image2pipe',      # Use image2pipe as the output format
    '-'
]

# Open the FFmpeg subprocess
process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, bufsize=10**6, shell=True)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Path for output video
# output_path = "Fire_Detect_v3.mp4"
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Create Video capture object
# video_cap = cv2.VideoCapture(video_path)

# Check if video opened
# if not video_cap.isOpened():
#     print("Error opening video")
#     exit()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Get video properties 
# if video_cap.isOpened():
#     width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = int(video_cap.get(cv2.CAP_PROP_FPS))

# Create video writer object - uncomment to save output video
# fourcc = cv2.VideoWriter.fourcc(*'DIVX')
# output = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Frame processing time collection array - Uncomment to calculate processing time
# processing_time = []
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    
# ///////////////////////////////////////////////////////////////////    
# Counter for frame extraction
# i = 0
# ///////////////////////////////////////////////////////////////////

while True:
    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    #start_time = time.time() # uncomment to calculate processing time
    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    raw_frame = process.stdout.read(1920 * 1080 * 3)  # Adjust the size based on your video resolution and pixel format
    
    if not raw_frame:
        break

    # Convert the raw frame to a NumPy array
    frame = np.frombuffer(raw_frame, dtype=np.uint8)
    frame = frame.reshape((1080, 1920, 3))  # Adjust the shape based on your video resolution and pixel format

    # Rescale Frame
    frame = rescaleFrame(frame)
    cv2.imshow('Original Frame', frame)

    # Call Image Processing Class and produce detection results
    ImgProc = ImageProcessor(G_Kernel, E_Kernel, D_Kernel, E_Iter, D_Iter, Cont_Size, Cont_Cmplx)
    frame = ImgProc.FireDetect(frame) # Provide "i" as parameter if using utility code
    cv2.imshow('Detections', frame)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Write frame with bounding boxes to output video - uncomment to save video with detections
    # output.write(frame)
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # Uncomment to calculate processing time for each frame
    # end_time = time.time()
    # processing_time.append(end_time-start_time)
    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    # Break statement
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Uncomment to print mean frame processing time
# print(np.mean(processing_time))
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

# Release the video writer object
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# output.release() # uncomment if saving output video
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Release the video capture object and close all windows
cv2.destroyAllWindows()
process.terminate()
