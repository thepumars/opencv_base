import cv2 as cv
import numpy as np

#canny edge detection

img = cv.imread('pic/operate/footballshooter.jpg',0)

#parameters minval maxval sobelsize booleanforfomula:True for sqrt
sobelx = cv.Sobel(img,cv.CV_64F,1,0)

# 1 noise refuse
# 2 gradient calc
# 3 check if localmax
# 4 check if valid
edges = cv.Canny(img,100,200)

cv.imshow('sobelx',sobelx)

cv.imshow('edges',edges)

cv.waitKey(0)

cv.destroyAllWindows()