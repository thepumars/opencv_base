#before looking for outline, remember threshold
#or the canny detection
import cv2 as cv
import numpy as np

#!!!ALWAYS remember that the object whose outline we need to look for needs to be white,
# and the background should be black
img = cv.imread('pic/operate/footballshooter.jpg')

imgray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

ret,thresh = cv.threshold(imgray,127,255,cv.THRESH_BINARY)

#contours_mode contours_approximate e.g. use 4 points to stand for a rectangle

contours,hierarchy = cv.findContours(thresh,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

dev = cv.drawContours(thresh,contours,-1,(0,255,0),1)

cv.imshow('window',dev)

cv.waitKey(0)

cv.destroyAllWindows()
# cnt = contours[4]
# cv.drawContours(img,[cnt],0,(0,255,0),3)

moment = cv.imread('pic/moment.png',0)
cv.imshow('moment',moment)
cv.waitKey(0)

cv.destroyAllWindows()
ret,thresh1  = cv.threshold(moment,160,255,cv.THRESH_BINARY_INV)
cv.imshow('thresh1',thresh1)
cv.waitKey(0)

cv.destroyAllWindows()
contours,hierarchy = cv.findContours(thresh1, 1, 2)
cnt = contours[0]
dev = cv.drawContours(moment,[cnt],0,(0,0,255),3)

cv.imshow('dev',dev)

cv.waitKey(0)

cv.destroyAllWindows()
print(cnt)
M = cv.moments(cnt)
print(M)

#area calc  or by the M00

area = cv.contourArea(cnt)

#True for the closed geometry 

#arclength calc

perimeter = cv.arcLength(cnt,True)

#using epsilon to approximate the shape
#the applications of Douglas-Peucker algorithm

# epsilon = 0.01*cv.arcLength(cnt,True)
# approx = cv.approxPolyDP(cnt,epsilon,True)

# dev = cv.drawContours(moment,[approx],0,(0,0,255),3)

# cv.imshow('dev',dev)

# cv.waitKey(0)

# cv.destroyAllWindows()

#vexHull 找最大凸度缺陷
#hull = cv.convexHull(cnt)

k = cv.isContourConvex(cnt)
print(k)

##以下是寻找最小和拟合图形的，不再此列出
#https://apachecn.github.io/opencv-doc-zh/#/docs/4.0.0/4.9-tutorial_py_contours

#mask and all pixels that make up the graph
# mask = np.zeros(imgray.shape,np.uint8)
# cv.drawContours(mask,[cnt],0,255,-1)
# pixelpoints = np.transpose(np.nonzero(mask))
# # pixelpoints = cv.findNonZero(mask)

#using the mask we can find max and min & its location

#and the points mean the most


#same time you can use some methods to match the shape

#also remember you can use hierarchy

