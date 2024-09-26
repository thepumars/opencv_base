import cv2 as cv
import numpy as np

# e1 = cv.getTickCount()

# e2 = cv.getTickCount()

# time = (e2-e1)/cv.getTickFrequency()

img1 = cv.imread('pic/operate/footballshooter.jpg')
e1 = cv.getTickCount()

for i in range(5,49,2):
    img1 = cv.medianBlur(img1,i)
e2 = cv.getTickCount()

time = (e2-e1)/cv.getTickFrequency()

print(time)

print(cv.useOptimized())

#when the size of the array or list is very small(1 or 2 size), python operation is faster
