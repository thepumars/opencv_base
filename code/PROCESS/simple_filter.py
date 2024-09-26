import cv2 as cv
import numpy as np

from matplotlib import pyplot as plt



img = cv.imread('pic/opencv.png')


#average filter below
kernel = np.ones((5,5),np.float32)/25
dst = cv.filter2D(img,-1,kernel)

cv.imshow('window',dst)

cv.waitKey(0)

cv.destroyAllWindows()

#mean  cv.blur and cv.boxFilter

blur = cv.blur(img,(5,5))

cv.imshow('window',blur)

cv.waitKey(0)

cv.destroyAllWindows()

#if you would like the rectangle kernel to process
#use boxFilter and explicitly extern that normalize = False

#gaussian filter

blur = cv.GaussianBlur(img,(5,5),0)

cv.imshow('window',blur)

cv.waitKey(0)

cv.destroyAllWindows()

#midvalue for salt and pepper noise


blur = cv.medianBlur(img,5)

cv.imshow('window',blur)

cv.waitKey(0)

cv.destroyAllWindows()

#bilateral filter
