

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

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
    img = cv2.imread(image)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    
    # If found, draw corners
    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)


print (imgpoints)
print(objpoints)
img = cv2.imread('camera_cal/calibration1.jpg')

# mtx  ==> camera matrix  used to transform 3D object points to 2D image points 
# dist ==> distortion coefficient
# camera position in the world 
#   rvecs rotation vector
#   tvecs translation vector
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

dst = cv2.undistort(img, mtx, dist, None, mtx)

f, (ax1, ax2) = plt.subplots(1, 2)
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(dst)
ax2.set_title('Undistorted Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)