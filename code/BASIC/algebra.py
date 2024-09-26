import cv2 as cv
import numpy as np

x = np.uint8([250])
y = np.uint8([10])
print(x,y)
print(cv.add(x,y))

print(x+y)

#mostly the func from the cv is more effective and reasonable,consider using
#native func more

# weight add
#cv.addWeighted(img1,0.7,img2,0.3)

#following well be talking about calc in bit or i will say mask

img1 = cv.imread('pic/operate/logo.jpg')
img2 = cv.imread('pic/operate/yellow.jpg')
rows,cols,channels = img2.shape
roi  = img1[0:rows,0:cols]

img2gray = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
ret,mask = cv.threshold(img2gray,10,255,cv.THRESH_BINARY)
mask_inv = cv.bitwise_not(mask)
img1_bg = cv.bitwise_and(roi,roi,mask=mask_inv)
img2_fg = cv.bitwise_and(img2,img2,mask=mask)
dst = cv.add(img1_bg,img2_fg)
img1[0:rows,0:cols] = dst
cv.imshow('res',img1)
cv.waitKey(0)
cv.destroyAllWindows()