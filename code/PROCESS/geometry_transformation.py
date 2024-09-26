import numpy as np
import cv2 as cv

#resize
# img = cv.imread('pic/operate/footballshooter.jpg')
# cv.imshow('window1',img)
# res = cv.resize(img,None,fx=2,fy=2,interpolation=cv.INTER_CUBIC)
# #OR
# #height,width = img.shape[:2]

# #res = cv.resize(img,(2*width,2*height),interpolation=cv.INTER_CUBIC)
# cv.imshow('window2',res)
# cv.waitKey(0)


#translate
img = cv.imread('pic/operate/footballshooter.jpg',0)
rows,cols = img.shape

M = np.float32([[1,0,100],[0,1,50]])
dst = cv.warpAffine(img,M,(cols,rows))

cv.imshow('img',dst)
cv.waitKey(0)
cv.destroyAllWindows()

mg = cv.imread('pic/operate/footballshooter.jpg',0)
rows,cols = img.shape

M = cv.getRotationMatrix2D((cols/2,rows/2),90,1)
dst = cv.warpAffine(img,M,(cols,rows))

cv.imshow('img',dst)
cv.waitKey(0)
cv.destroyAllWindows()

img = cv.imread('pic/sudoku.png')

rows,cols,ch = img.shape

pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])

#not the original picture,sol: select point by click the button 
#and then make the transformation


# M = cv.getPerspectiveTransform(pts1,pts2)

# dst = cv.warpPerspective(img,M,(300,300))

# cv.imshow()
