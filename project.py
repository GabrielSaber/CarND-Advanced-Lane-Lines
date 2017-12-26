import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob


'''
globale variables
'''
# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

'''
brief   funtion to calibrate the camera

input   nothing 

return  mtx  ==> camera matrix  used to transform 3D object points to 2D image points
        dist ==> distortion coefficient    
'''
def calibrate_camera():
    # prepare object points
    nx = 9#TODO: enter the number of inside corners in x
    ny = 5#TODO: enter the number of inside corners in y
    
    # Make a list of calibration images
    images = glob.glob('camera_cal/calibration*.jpg')
    
    # Find the chessboard corners
    objpoints = []
    imgpoints = []
    
    objp = np.zeros((5*9, 3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:5].T.reshape(-1, 2)
    
    for image in images:
        img = mpimg.imread(image)
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        # If found, draw corners
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)
            
    # mtx  ==> camera matrix  used to transform 3D object points to 2D image points 
    # dist ==> distortion coefficient
    # camera position in the world 
    #   rvecs rotation vector
    #   tvecs translation vector
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return mtx, dist

'''
brief   funtion to undistort image

input   distorted_img ==> distorted image
        mtx  ==> camera matrix  used to transform 3D object points to 2D image points
        dist ==> distortion coefficient

return  undistorted image
'''
def undistort_image(distorted_img, mtx, dist):
    dst = cv2.undistort(distorted_img, mtx, dist, None, mtx)
    return dst

'''
brief   funtion to plot two images on x axis

input   img1 ==> first image
        img2 ==> second image
        text1 ==> first text
        text1 ==> seconf test
'''
def plot(img1, img2, text1='image_1', text2 ='image_2'):
    f, (ax1, ax2) = plt.subplots(1, 2)
    f.tight_layout()
    ax1.imshow(img1)
    ax1.set_title(text1, fontsize=20)
    ax2.imshow(img2)
    ax2.set_title(text2, fontsize=20)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    
'''
brief   funtion to get the combined binary image from color threshold and gradient threshold 

input   undist_img ==> undistored image
        color_thrshld ==> second image
        grad_thrshld ==> first text
        
return combined_binary ==> combined binary image from color threshold and gradient threshold
'''
def color_gradient_combined_binary(undist_img, color_thrshld=(170, 255), grad_thrshld=(20, 100)):
    # Convert to HLS color space and separate the S channel
    # Note: img is the undistorted image
    hls = cv2.cvtColor(undist_img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    
    # Grayscale image
    # NOTE: we already saw that standard grayscaling lost color information for the lane lines
    # Explore gradients in other colors spaces / color channels to see what might work better
    gray = cv2.cvtColor(undist_img, cv2.COLOR_RGB2GRAY)
    
    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= grad_thrshld[0]) & (scaled_sobel <= grad_thrshld[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= color_thrshld[0]) & (s_channel <= color_thrshld[1])] = 1
    
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    
    return combined_binary


'''
brief   funtion to warp the image to bird eye view
input   img ==> image to warp
return  warped ==> warped image 
        M ==> warp matrix
        Minv ==> warp inverse matrix
'''
def warp_image_topview(img):
    img_size = img.shape[::-1]
    src = np.float32([[580, 460],
                      [701, 460],
                      [1037, 676],
                      [266,676]])

    dst = np.float32([[266, 160],
                      [1037, 160],
                      [1037, 676], 
                      [266, 676]])
    
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    
    Minv = cv2.getPerspectiveTransform(dst, src)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size)

    return warped, M, Minv

'''
brief   funtion to detect lane in first frame

input   binary_warped_img ==> binary wraped bird eye image

return  left_fit ==> left lane
        right_fit ==> right lane
'''
def find_lanes_first_frame(binary_warped_img):
    
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped_img[np.int32(binary_warped_img.shape[0]/2):,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped_img, binary_warped_img, binary_warped_img))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped_img.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped_img.nonzero()
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
        win_y_low = binary_warped_img.shape[0] - (window+1)*window_height
        win_y_high = binary_warped_img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
        (0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
        (0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    '''
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped_img.shape[0]-1, binary_warped_img.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    '''
    
    return left_fit, right_fit

'''
brief   funtion to detect lane in first frame

input   binary_warped_img ==> binary wraped bird eye image
        left_fit ==> left lane found from pervious frame
        right_fit ==> right lane found from pervious frame

return  ploty ==> 
        left_fitx ==> x values for plotting
        right_fitx ==> y values for plotting
'''
def find_lanes_next_frame(binary_warped, left_fit, right_fit):    
    # Assume you now have a new warped binary image 
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
    left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
    right_fit[1]*nonzeroy + right_fit[2] + margin)))  
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
     
    '''
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                                  ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                                  ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
   
    plt.imshow(result)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    '''
    return ploty, left_fitx, right_fitx
    
'''
brief   funtion to calulate lane curveture

input   ploty ==> 
        left_fitx ==> x values for plotting
        right_fitx ==> y values for plotting
        
retuen  left_curverad ==> left lane curveture radius
        right_curverad ==> right lane curveture radius
'''
  
def calculate_curveture(ploty, leftx, rightx):
    
    y_eval = np.max(ploty)
    
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    return left_curverad, right_curverad

'''
brief   funtion to draw lane polygon on the image and print lane curveture text

input   undist_img ==> input undistorted image
        binary_warped ==> binary wraped bird eye image
        Minv ==> Minv ==> warp inverse matrix
        ploty ==> 
        left_fitx ==> left lane points
        right_fitx ==> right lane points
        
retuen  result ==> output image with detected lane polygon and curveture text
       
'''
def draw(undist_img, binary_warped, Minv, ploty, left_fitx, right_fitx):
    
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist_img.shape[1], undist_img.shape[0])) 

    left_curverad, right_curverad = calculate_curveture(ploty, left_fitx, right_fitx)
    
    lane_curverad = (left_curverad + right_curverad)/2

    cv2.putText(undist_img,'Radius of curvature  = %.2f m'%(lane_curverad),(50,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)
    
    lane_center =  (right_fitx[-1] - left_fitx[-1])/2 + left_fitx[-1]
    car_pos = undist_img.shape[1]/2 
    car_pos_wrt_lane_center = (car_pos - lane_center)*xm_per_pix
   
    text = "Vehicle is %.2fm " % abs(car_pos_wrt_lane_center)
    
    if car_pos_wrt_lane_center < 0:
        text = text + "left of the center"
        
    else:
        text = text + "right of the center"
        
    cv2.putText(undist_img,text,(50,100), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)
   
    
    # Combine the result with the original image
    result = cv2.addWeighted(undist_img, 1, newwarp, 0.3, 0)
    
    #plt.imshow(result)
    return result
    

'''
brief   the process image function

input   img == > inputimage
    
retuen  result ==> output image with detected lane polygon and curveture text
       
'''
def process_image(image):
    dst = undistort_image(image, mtx, dist)
    binary_img = color_gradient_combined_binary(dst)
    binary_warped, M, Minv = warp_image_topview(binary_img)
    left_fit_first, right_fit_first = find_lanes_first_frame(binary_warped)
    ploty, left_fitx_next, right_fitx_next = find_lanes_next_frame(binary_warped, left_fit_first, right_fit_first)
    result = draw(dst, binary_warped, Minv, ploty, left_fitx_next, right_fitx_next)
    return result
    
#calibrate camera
mtx, dist = calibrate_camera()

#test image
im = mpimg.imread("test_images/straight_lines1.jpg")
result = process_image(im)
plt.imshow(result)


# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip

'''
import and process video
'''
white_output = 'output_video/output_video.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(process_image)
white_clip.write_videofile(white_output, audio=False)

