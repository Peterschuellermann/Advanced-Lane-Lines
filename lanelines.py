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

def process_image(image):

    # Distortion Correction
    undistorted = cv2.undistort(image, mtx, dist, None, mtx)

    # Perspective Transform
    warped = warp_image(undistorted, warp_matrix)

    return warped





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