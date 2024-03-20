# SWARMSThermal
The setup for this project is as follows:
- Python 3.7
- OpenCV 4.2 (requires opencv-python installation as well - using conda/PyPI)
- Numpy
- Subprocess
- FFMPEG

OpenCV installation guide: https://www.youtube.com/watch?v=trXs2r6xSnI&t=69s
FFMPEG installation guide:

Numpy and Subprocess can be installed through regular python package managers like conda or PyPI

The final version of the project is organised in the ThermalVideoDetection_Final folder. The file that you need to run is the filename that ends in "main". Ensure that you have the "ImgProcessor" and "util" files in the same folder as the "main" file as they are essential to the functioning of the script. Read through the comments in the scripts as a guide for changing the code should you need to.

Simple image processing technique for detecting hot spots in "white-hot" thermal imagery and videos. Algorithm delivers real-time processing for videos.
Steps involved in processing are:
1) Frame-by-Frame processing for videos
2) Conversion of the frame to grayscale
3) Extracting the mean pixel intensity and the standard deviation
4) Setting the image binarising threshold to mean+(2xstd.dev) *(If threshold exceeds maximum value of 255, the threshold is brought down to mean+(1xstd.dev))*
5) Performing Gaussian Blurring on the image *(with tunable kernel size and thresholds)*
6) Binarising the image with previously determined threshold
7) Performing Erosion operation on the image *(with tunable parameters)*
8) Performing Dilation operation on resultant image *(with tunable parameters)*
9) Drawing contours on resultant image
10) Drawing bounding boxes on detected contours *(with two tunable parameters, namely: contour complexity and contour area)*
