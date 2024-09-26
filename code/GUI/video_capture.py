import numpy as np

import cv2 as cv

cap = cv.VideoCapture(0)

if(~cap.isOpened()):
    cap.open(0)
   
print('{0} x {1} '.format(cap.get(cv.CAP_PROP_FRAME_WIDTH),cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
cap.set(cv.CAP_PROP_FRAME_WIDTH,320)
cap.set(cv.CAP_PROP_FRAME_HEIGHT,240)
# you can use the cap get to have the parameter of the video
# use set to set the value
while(True):
    ret,frame = cap.read() # ret is a bool value to show if the cap is ok

    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY) #converter
    cv.imshow('image',gray)
    if cv.waitKey(16) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()



