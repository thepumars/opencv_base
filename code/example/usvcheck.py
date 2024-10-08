import cv2 as cv
import numpy as np

hsv = np.array([[[254,91,100]]],dtype=np.uint8)

print(hsv.shape)
def hsv2bgr(hsv):
    tmp = hsv
    tmp[0,0,0] = tmp[0,0,0]/2
    tmp[0,0,1] = int(tmp[0,0,1]*2.56)
    tmp[0,0,2] = tmp[0,0,2]
    
    bgr = cv.cvtColor(tmp,cv.COLOR_HSV2BGR)

    return bgr

bgr = hsv2bgr(hsv)

print(bgr)