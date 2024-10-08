import cv2 as cv

import numpy as np

import matplotlib.pyplot as plt

from itertools import permutations


#create a blank image 800x450

blank = np.zeros((450,800))


plt.figure('blank images')




perms = np.array(list(permutations(range(5), 5)))
perms = perms+1

zero_add = np.zeros((120,1),dtype=np.uint8)

perms_new = np.column_stack((zero_add,perms))
perms_new = perms_new+1
print(perms_new)
# [[379  77]
#  [420  74]
#  [414  93]
#  [423 111]
#  [381 114]
#  [380  95]]




dots = [[379,77],[420,74],[414,93],[423,111],[381,114],[380,95]]
dots = np.array(dots)
order =np.array([1,3,5,2,4,6])
order =order -1
# dots = dots[order]
point_size = 1
point_color = (255, 255, 255) # BGR
thickness = -1 #  0 4 8
for i,perm in enumerate(perms_new):
    blank = np.zeros((450,800))
    dot = dots[perm-1]
    polylines_img = cv.polylines(blank,[dot],True,(255,255,255))
    
    cv.putText(blank,str(perm),(400,300),cv.FONT_HERSHEY_SIMPLEX,1,(255, 255, 255), 2)
    cv.imshow('pic{}'.format(i),blank)
    # cv.putText(img,'OpenCv',(128,400),cv.FONT_HERSHEY_COMPLEX_SMALL,2,(255,255,255),8,lineType=cv.LINE_AA)
    while chr(cv.waitKey(0)) != 's':
        pass
    cv.destroyWindow('pic{}'.format(i))
        




for pointx,pointy in dots:
	cv.circle(blank, (pointx,pointy), point_size, point_color, thickness)



# dot above

plt.imshow(blank)

plt.show()