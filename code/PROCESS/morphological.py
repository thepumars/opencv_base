import numpy as np
import cv2 as cv

img = cv.imread('pic/morph.png')

blur = cv.blur(img,(5,5))

#tested: when kernel all zeros nothing changed
#calling the function of erode
kernel = np.ones((5,5),np.uint8)


cv.imshow('img',img)

cv.waitKey(0)

cv.destroyAllWindows()

dst = cv.erode(img,kernel)

cv.imshow('erode',dst)

cv.waitKey(0)

cv.destroyAllWindows()
#iterations = 1 means the cnt of opr
dst = cv.dilate(img,kernel)

cv.imshow('dilate',dst)

cv.waitKey(0)

cv.destroyAllWindows()

#open opr erode then dilate to delete the dot round

img = cv.imread('pic/open.png')

opening = cv.morphologyEx(img,cv.MORPH_OPEN,kernel)

cv.imshow('opening',opening)

cv.waitKey(0)

cv.destroyAllWindows()

#close opr dilate then erobe to delete the holes

img = cv.imread('pic/close.png')

closing = cv.morphologyEx(img,cv.MORPH_CLOSE,kernel)

cv.imshow('closing',closing)

cv.waitKey(0)

cv.destroyAllWindows()

#gradient  to draw the outline dilate - erode
img = cv.imread('pic/morph.png')

gradient = cv.morphologyEx(img,cv.MORPH_GRADIENT,kernel)

cv.imshow('gradient',gradient)

cv.waitKey(0)

cv.destroyAllWindows()

#tophat original - open kernel 9x9
#example increase the light of front font e.g. the visa card number
#  

kernel = np.ones((9,9))

img = cv.imread('pic/morph.png')

tophat = cv.morphologyEx(img,cv.MORPH_TOPHAT,kernel)

cv.imshow('tophat',tophat)

cv.waitKey(0)

cv.destroyAllWindows()
#blackhat original - close kernel 9x9
kernel = np.ones((9,9))

img = cv.imread('pic/morph.png')

blackhat = cv.morphologyEx(img,cv.MORPH_BLACKHAT,kernel)

cv.imshow('blackhat',blackhat)

cv.waitKey(0)

cv.destroyAllWindows()