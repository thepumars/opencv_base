import os
envpath = '/home/mars/anaconda3/lib/python3.11/site-packages/cv2/qt/plugins'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath
import numpy as np

import cv2 as cv

from matplotlib import pyplot as plt

img = cv.imread('pic/test.jpg',1)

# plt.imshow(img,cmap = 'gray',interpolation='bicubic')

plt.xticks([]),plt.yticks([])
plt.show()