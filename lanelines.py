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
            print(".")
            # Draw and display the corners
            # img = cv2.drawChessboardCorners(img, (8, 6), corners, ret)

    return cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


ret, mtx, dist, rvecs, tvecs = calibration()

# image = cv2.cvtColor(mpimg.imread(images[0]), cv2.COLOR_BGR2GRAY)
# dst = cv2.undistort(image, mtx, dist, None, mtx)


# Distortion Correction

# Perspective Transform

# Detect Lane Pixels

# Determine Curvature

# Paint Curvature onto the original image

# Calculate estimation of the curve radius and print to video