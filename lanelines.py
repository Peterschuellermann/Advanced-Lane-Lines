# Peter Schüllermann
# Udacity Lane Lines Project for the CarND 2017
from moviepy.editor import VideoFileClip
import numpy as np
import cv2
import matplotlib.image as mpimg
import glob


def calibration():
    images = glob.glob("camera_cal/calibration*.jpg")

    imgpoints = []
    objpoints = []

    objp = np.zeros((9 * 6, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)  # x, y coordinates

    for fname in images:
        img = mpimg.imread(fname)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)

    return cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


def generate_warp_config():
    # source for corner points: https://github.com/js1972
    corners = np.float32([[253, 697], [585, 456], [700, 456], [1061, 690]])
    new_top_left = np.array([corners[0, 0], 0])
    new_top_right = np.array([corners[3, 0], 0])
    offset = [50, 0]

    src = np.float32([corners[0], corners[1], corners[2], corners[3]])
    dst = np.float32([corners[0] + offset, new_top_left + offset, new_top_right - offset, corners[3] - offset])

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

    return combined_binary


def detect_lane_lines(image):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(image[360:, :], axis=0)

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((image, image, image))  # * 255

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9

    # Set height of windows
    window_height = np.int(image.shape[0] / nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100

    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = image.shape[0] - (window + 1) * window_height
        win_y_high = image.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
            nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
            nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    return image, left_lane_inds, nonzerox, nonzeroy, out_img, right_lane_inds


def calculate_radius_and_center(image, left_lane_inds, nonzerox, nonzeroy, out_img, right_lane_inds):

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # RADIUS
    # calculate curve radius
    y_eval = np.max(ploty)
    left_fit = np.polyfit(lefty, leftx, 2)
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fit = np.polyfit(righty, rightx, 2)
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30.0 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    # calculate center displacemet
    left_distance = (left_fitx[719] * xm_per_pix)
    right_distance = (right_fitx[719] * xm_per_pix)
    center = (1280/2)*xm_per_pix
    center_distance = center - (left_distance + right_distance)/2

    return left_fitx, ploty, right_fitx, left_curverad, right_curverad, center_distance


def draw_lines_to_image(image, left_fitx, original, ploty, right_fitx):

    warp_zero = np.zeros_like(image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    new_warp = cv2.warpPerspective(color_warp, warp_matrix_inverse, (image.shape[1], image.shape[0]))

    # Combine the result with the original image
    result = cv2.addWeighted(original, 1, new_warp, 0.3, 0)
    return result


def write_info_to_image(image, left_curverad, right_curverad, center_distance):
    font = cv2.FONT_HERSHEY_SIMPLEX
    curve_radius = (left_curverad + right_curverad)/2

    # print curve radius
    if curve_radius >= 3000.0:
        cv2.putText(image, 'straight',
                    (320, 40), font, 1, (255, 255, 255), 2)
    else:
        cv2.putText(image, 'curve radius = ' + "{0:.2f}".format(round(curve_radius, 2)) + 'm',
                (320, 40), font, 1, (255, 255, 255), 2)

    # print center distance
    cv2.putText(image, 'center distance = ' + "{0:.2f}".format(round(center_distance, 2)) + 'm',
                (320, 100), font, 1, (255, 255, 255), 2)
    return image


def process_image(image):
    original = image

    # Distortion Correction
    image = cv2.undistort(image, mtx, dist, None, mtx)

    # Convert to HLS and detect Lane Pixels
    image = HLS_Gradient(image)

    # Perspective Transform
    image = warp_image(image, warp_matrix) * 255

    # detect lane lines
    image, left_lane_inds, nonzerox, nonzeroy, out_img, right_lane_inds = detect_lane_lines(image)

    # calculate curve radius and center displacement
    left_fitx, ploty, right_fitx, left_curverad, right_curverad, center_distance = \
        calculate_radius_and_center(image, left_lane_inds, nonzerox, nonzeroy, out_img, right_lane_inds)

    # draw area to image
    image = draw_lines_to_image(image, left_fitx, original, ploty, right_fitx)

    # write curve radius and center displacement to image
    image = write_info_to_image(image, left_curverad, right_curverad, center_distance)

    return image


ret, mtx, dist, rvecs, tvecs = calibration()
print("Generated calibration data!")

warp_matrix, warp_matrix_inverse = generate_warp_config()

test_image = mpimg.imread("test_images/test3.jpg")
test_image = process_image(test_image)
mpimg.imsave("test_image.jpg", test_image)

# video = VideoFileClip("project_video.mp4")
# video_processed = video.fl_image(process_image)
# video_processed.write_videofile("project_output.mp4", audio=False)


