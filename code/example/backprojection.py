import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

test = cv.imread('pic/2.jpg')
roi = test[80:92,385:398,:]
target = cv.cvtColor(test,cv.COLOR_BGR2HSV)
hsv = cv.cvtColor(roi,cv.COLOR_BGR2HSV)
# calculating object histogram
roihist = cv.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )
plt.figure('eigen pic')
plt.imshow(roihist,interpolation='nearest')
plt.colorbar()
###threshold the graph
##Need to adjust the sample size of ROI or extract sample features of lights to make the coverage area more certain
ret,roihist_threshold = cv.threshold(roihist,10000,10000,cv.THRESH_BINARY_INV)

roihist_new = cv.bitwise_and(roihist,roihist_threshold)

# print(hist2d_threshold)
# hist2d_tmp = hist2d_new[hist2d_new!=0]
# np.savetxt("data/threshold.csv", hist2d_tmp, delimiter=',')

# plt.figure('hist2d_threshold')

# plt.imshow(hist2d_threshold)

plt.figure('roihist_new')

plt.imshow(roihist_new)
plt.colorbar()
# normalize histogram and apply backprojection
cv.normalize(roihist_new,roihist_new,0,255,cv.NORM_MINMAX)
plt.figure('roihist_normalize')
plt.imshow(roihist_new)
plt.colorbar()
plt.show()
dst = cv.calcBackProject([target],[0,1],roihist_new,[0,180,0,256],1)
# Now convolute with circular disc
disc = cv.getStructuringElement(cv.MORPH_RECT,(5,5))
cv.filter2D(dst,-1,disc,dst)
# threshold and binary AND
ret,thresh = cv.threshold(dst,100,255,0)
thresh = cv.merge((thresh,thresh,thresh))
res = cv.bitwise_and(target,thresh)
res = np.vstack((target,thresh,res))
cv.imwrite('res/backprojection.jpg',res)
