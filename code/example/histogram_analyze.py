##histogram to analyze the status of the gray picture

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt



test = cv.imread('pic/2.jpg')

h,w,c = test.shape 

#waring in cv bgr = rgb in matplotlib
print("------------basic-----------info--------------")
print("figsize: {}x{} ,channels: {}".format(w,h,c))

test_plt = test[:,:,::-1]

plt.figure('test')
plt.imshow(test_plt)

roi = test[80:92,385:398,:]
roi_plt = roi[:,:,::-1]
plt.figure('roi_plt')
plt.imshow(roi_plt)

histroi = cv.calcHist([roi],[0,1],None,[180,256],[0,180,0,256])
plt.figure('histroi')
plt.imshow(histroi,interpolation='nearest')
plt.colorbar()

gray = cv.cvtColor(test,cv.COLOR_BGR2GRAY)


print("shape :",gray.shape)

plt.figure('gray')
plt.imshow(gray,cmap=plt.cm.gray)

test_hsv = cv.cvtColor(test,cv.COLOR_BGR2HSV)
plt.figure('test_hsv')
test_hsv_plt = test_hsv[:,:,::-1]

plt.imshow(test_hsv)

hist2d = cv.calcHist([test_hsv],[0,1],None,[180,256],[0,180,0,256])

# seq = np.sort(hist2d.ravel())

# seq_new = np.array([x for x in seq if x != 0])

# np.savetxt("data/new.csv", seq_new, delimiter=',')

# plt.figure('hist2D')


plt.imshow(hist2d,interpolation='nearest')

###threshold the graph

ret,hist2d_threshold = cv.threshold(hist2d,500,500,cv.THRESH_BINARY_INV)

hist2d_new = cv.bitwise_and(hist2d,hist2d_threshold)

# print(hist2d_threshold)
# hist2d_tmp = hist2d_new[hist2d_new!=0]
# np.savetxt("data/threshold.csv", hist2d_tmp, delimiter=',')

# plt.figure('hist2d_threshold')

# plt.imshow(hist2d_threshold)

plt.figure('hist2d_new')

plt.imshow(hist2d_new,cmap=plt.cm.gray)

print("hsv-hist-shape: ",hist2d.shape)


plt.colorbar()

#normalize

cv.normalize(hist2d,hist2d,0,255,cv.NORM_MINMAX)


plt.figure('hist1D')



# cv.calcBackProject(test_hsv,,hist2d,[180,256],1)

plt.hist(gray.ravel(),256,[0,256])

plt.show()