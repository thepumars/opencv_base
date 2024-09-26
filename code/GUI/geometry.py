import cv2 as cv

import numpy as np

geo_path = 'pic/geo.jpg'

img = np.zeros((512,512,3),dtype=np.uint8)

#using the numpy to make a pic for use

cv.line(img,(0,0),(511,511),(255,0,0),5)

#dot cordinate and color and width

cv.rectangle(img,(384,0),(510,128),(0,255,0),2)

#dot cordinate and color and width

cv.circle(img,(447,63),63,(0,0,255),-1)

cv.ellipse(img,(256,256),(100,50),0,45,200,255,-1)

pts = np.array([[10,5],[20,30],[70,20],[50,10]],np.int32)

pts=pts.reshape((-1,1,2))

cv.putText(img,'OpenCv',(128,400),cv.FONT_HERSHEY_COMPLEX_SMALL,2,(255,255,255),8,lineType=cv.LINE_AA)

cv.polylines(img,[pts],True,(0,255,255))


cv.imshow('window',img)

#when call the func in terminal err occurs

cv.imwrite(geo_path,img)

cv.destroyAllWindows()
