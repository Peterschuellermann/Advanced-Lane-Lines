# Peter Schüllermann
# Udacity Lane Lines Project for the CarND 2017
from moviepy.editor import VideoFileClip
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import math # for slope calculations


# Camera Calibration
def calibration():

    images = glob.glob("camera_cal/calibration*.jpg")

    imgpoints = []
    objpoints = []

    objp = np.zeros((9 * 6, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)  # x, y coordinates

    for fname in images:
        img = mpimg.imread(fname)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)

    return cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

def generate_warp_config():


    src = np.float32([[0, 670], [1280, 670], [0, 450], [1280, 450]])
    dst = np.float32([[570, 220], [710, 220], [0, 0], [1280, 0]])

    warp_matrix = cv2.getPerspectiveTransform(src, dst)
    warp_matrix_inverse = cv2.getPerspectiveTransform(dst, src)
    return warp_matrix, warp_matrix_inverse


def warp_image(image, warp_matrix):
    img_size = (image.shape[1], image.shape[0])

    warped = cv2.warpPerspective(image, warp_matrix, img_size, flags=cv2.INTER_LINEAR)

    return warped


def HLS_Gradient(image):
    # Convert to HLS color space and separate the S channel
    # Note: img is the undistorted image
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    s_channel = hls[:, :, 2]

    # Grayscale image
    # NOTE: we already saw that standard grayscaling lost color information for the lane lines
    # Explore gradients in other colors spaces / color channels to see what might work better
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    thresh_min = 20
    thresh_max = 100
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Threshold color channel
    s_thresh_min = 170
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return color_binary


def process_image(image):

    # Distortion Correction
    image = cv2.undistort(image, mtx, dist, None, mtx)


    # Perspective Transform
    image = warp_image(image, warp_matrix)


    # Convert to HLS
    # Detect Lane Pixels
    image = HLS_Gradient(image)


    return image





ret, mtx, dist, rvecs, tvecs = calibration()
print("Generated calibration data!")
warp_matrix, warp_matrix_inverse = generate_warp_config()


video = VideoFileClip("project_video.mp4")

video_processed = video.fl_image(process_image) #NOTE: this function expects color images!!

video_processed.write_videofile("project_output.mp4", audio=False)





# Detect Lane Pixels

# Determine Curvature

# Paint Curvature onto the original image

# Calculate estimation of the curve radius and print to video