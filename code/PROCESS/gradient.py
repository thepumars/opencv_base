import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
ret1 = cv.imread('pic/sudoku.png',0)
# Otsu's thresholding
#ret1 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)

laplacian = cv.Laplacian(ret1,cv.CV_64F)
sobelx = cv.Sobel(ret1,cv.CV_64F,1,0,ksize=5)
sobely = cv.Sobel(ret1,cv.CV_64F,0,1,ksize=5)
img_x = cv.convertScaleAbs(sobelx)
img_y = cv.convertScaleAbs(sobely)

mask = cv.addWeighted(img_x,.5,img_y,.5,1)

cv.imshow('laplacian',laplacian)

cv.waitKey(0)

cv.destroyAllWindows()
# plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
# plt.title('Original'), plt.xticks([]), plt.yticks([])
# plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
# plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
# plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
# plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
# plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
# plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
# plt.show()


