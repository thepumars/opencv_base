import cv2 as cv
import numpy as np

#generate one yellow picture

# image = np.zeros((128,128,3),dtype=np.uint8)
# image[:,:,1] = 255

# image[:,:,2] = 255

# cv.imshow('img',image)
# k = cv.waitKey(0)
# if k == 27:
#     cv.destroyAllWindows()
# elif k == ord('s'):
#     cv.imwrite('pic/operate/yellow.jpg',image)
#     cv.destroyAllWindows()

img = cv.imread('pic/operate/yellow.jpg')
h,w,c = img.shape
print(h,w,c)

print(img[100,100])

blue = img[100,100,0]
 
img[100,100]=[255,255,0]

print(img[100,100])

cv.imshow('window',img)
cv.waitKey(0)

#it's not recommended that accessing the value of single pixel to change it
#try using item and itemset to do this

print(img.shape)
print(img.size)
print(img.dtype)

#considering the roi region of interesting

#using slice of the numpy to do this

#split and merge the picture

b,g,r = cv.split(img)

cv.imshow('blue',b)

cv.waitKey(0)

img = cv.merge((b,g,r))

cv.destroyAllWindows()

#remember that split and merge are valuable operations,normally 
#we use numpy to get the slices

#border maker

cv.copyMakeBorder(img,10,10,10,10,cv.BORDER_CONSTANT,value=200)
cv.copyMakeBorder(img,10,10,10,10,cv.BORDER_REFLECT)
cv.copyMakeBorder(img,10,10,10,10,cv.BORDER_REFLECT_101)
cv.copyMakeBorder(img,10,10,10,10,cv.BORDER_REPLICATE)
cv.copyMakeBorder(img,10,10,10,10,cv.BORDER_WRAP)

