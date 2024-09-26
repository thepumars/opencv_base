import cv2 as cv
print(cv.__version__)

exam_path = 'pic/test.jpg'

cv.imread(exam_path,cv.IMREAD_COLOR)  #second parameter set how to read (gray or some)

#flags 1 0 -1 stands for the color gray and unchanged(alpha)

img = cv.imread(exam_path,0)

#when path wrong nothing err thrown out but none when print

cv.imshow('show',img)

cv.waitKey(0) # 0 is the number for milisec to wait

cv.destroyAllWindows()

cv.namedWindow('image',cv.WINDOW_AUTOSIZE)
#seems like the figure and add subplot   window ahead the picture
#normally autosize is the default setting
#when set to normal the size of window can be adjusted
cv.imshow('image',img)

cv.waitKey(0)

cv.destroyAllWindows()

cv.imwrite('save.jpg',img)
