import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


img = cv.imread('pic/test.jpg',0)
# create a CLAHE object (Arguments are optional).
clahe = cv.createCLAHE(clipLimit=.5, tileGridSize=(3,3))
res1 = clahe.apply(img)
res2 = cv.equalizeHist(img)

res = np.vstack((img,res1,res2))

plt.imshow(res,cmap=plt.cm.gray)

plt.show()

cv.imwrite('res/img_balanced.jpg',res)

