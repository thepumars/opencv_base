import cv2 as cv
import numpy as np

from matplotlib import pyplot as plt

img = cv.imread('pic/gradient.png')

ret,thresh1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
ret,thresh2 = cv.threshold(img,127,255,cv.THRESH_BINARY_INV)
ret,thresh3 = cv.threshold(img,127,255,cv.THRESH_TRUNC)
ret,thresh4 = cv.threshold(img,127,255,cv.THRESH_TOZERO)
ret,thresh5 = cv.threshold(img,127,255,cv.THRESH_TOZERO_INV)

cv.imshow('window',thresh2)
cv.waitKey(0)
cv.destroyAllWindows()

#adaptive threshold image

img = cv.imread("pic/sudoku.png",0)

th1 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,2)
th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
ret,th3 = cv.threshold(img,127,255,cv.THRESH_BINARY)


cv.imshow('window1',img)
cv.imshow('window2',th1)
cv.imshow('window3',th2)
cv.imshow('window4',th3)
cv.waitKey(0)
cv.destroyAllWindows()

# the retValue above is aiming to help you
#manipulate the pic with otsu
#https://cloud.tencent.com/developer/article/2068739

'''
# global thresholding
ret1,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)

# Otsu's thresholding
ret2,th2 = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

# Otsu's thresholding after Gaussian filtering
blur = cv.GaussianBlur(img,(5,5),0)
ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
'''
#大津法