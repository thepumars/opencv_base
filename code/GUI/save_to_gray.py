import cv2 as cv
import numpy as np

img = cv.imread('pic/test.jpg',0)
cv.imshow('image',img)

k  = cv.waitKey(0)

if k == 27:
    cv.destroyAllWindows()

elif k == ord('s'):
    cv.imwrite('pic/gray.jpg',img)
    cv.destroyAllWindows()
else:
    print(k)

#careful the tutorials told that err may occur when using x64 machine, 
#that time need to & 0xff
