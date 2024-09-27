import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
# img = cv.imread('pic/home.png',0)

# #using plt.hist you dont need to use calcHist
# plt.hist(img.ravel(),256,[0,256]); plt.show()

#for colorful graph using calc to find and plt to draw

# B G R hist draw

# img = cv.imread('pic/home.png')
# colors = ('b','g','r')
# plt.figure()
# for i,col in enumerate(colors):
#     hist = cv.calcHist([img],[i],None,[256],[0,256])
#     plt.plot(hist,color=col)
#     plt.xlim([0,256])
# plt.show()

#normal balance 

# cv.equalizeHist

# adaptive histogram balance

# img = cv.imread('tsukuba.png',0)
# # create a CLAHE object (Arguments are optional).
# clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# cl1 = clahe.apply(img)
# cv.imwrite('clahe_2.jpg',cl1)

#here comes the 2D histogram

# img = cv.imread('pic/home.png')

# hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)

# hist = cv.calcHist([hsv],[0,1],None,[180,256],[0,180,0,256])

# #using cv.imshow will express by gray graph
# plt.figure()
# plt.imshow(hist,interpolation='nearest')
# plt.show()

#back projection based on histogram

