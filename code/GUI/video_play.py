import numpy as np
import qtpy
import cv2 as cv

videopath = 'video/test.mp4'
cap = cv.VideoCapture(videopath)
while(cap.isOpened()):
    ret,frame =  cap.read()
    if ret == True:
        gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        cv.imshow('frame',frame)

        if cv.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()

