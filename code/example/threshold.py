#encoding=utf-8
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sg

def get_maxima(values: np.ndarray):

    max_index = sg.argrelmax(values)[0]
    return max_index, values[max_index]

test = cv.imread('pic/3.jpg')
test_plt = test[:,:,::-1]



gray = cv.cvtColor(test,cv.COLOR_BGR2GRAY)

zero_like_mat = np.zeros_like(gray)

blur = cv.GaussianBlur(gray,(5,5),0)

laplacian = cv.Laplacian(gray,cv.CV_64F)

# plt.figure('gaussian blur')

plt.figure('laplacian blur')

res = np.vstack((blur,laplacian))

plt.imshow(res,cmap=plt.cm.gray)




hist = cv.calcHist([gray],[0],None,[256],[0,256])

ret2,th2 = cv.threshold(gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
plt.figure('traditional Otcus')

plt.imshow(th2,cmap=plt.cm.gray)

# np.savetxt('data/hist.csv',hist,delimiter=',')

# print(hist)

max_index,values = get_maxima(hist)

index = np.argsort(values,axis=0)

flag = 1

tmp = index[flag][0]

# print(tmp)

ret3,th3 = cv.threshold(gray,int(tmp),255,cv.THRESH_BINARY)

plt.figure('main component1')

plt.imshow(th3,cmap=plt.cm.gray)

designed_method = gray[gray > tmp]

hist_new = cv.calcHist([designed_method],[0],None,[256],[0,256])

# np.savetxt('data/designed_method.csv',hist_new,delimiter=',')

# print(hist_new)

pdf = hist_new / designed_method.shape

# print(pdf)

plt.figure("pdf")

plt.plot(pdf)

cdf = pdf.cumsum()

plt.plot(cdf)

sequence = np.arange(0,256)

pdf = pdf[:,0]

Expectation = pdf*sequence

# print(Expectation)


# np.savetxt('data/Expectation.csv',Expectation,delimiter=',')
eps = np.finfo(Expectation.dtype).eps
eps_sequence =  eps*np.ones(256)
square_err = np.divide(np.square(Expectation[255]*pdf-Expectation),(pdf-np.square(pdf)+eps))
# print(square_err)

# np.savetxt('data/numerator.csv',np.square(Expectation[255]*pdf-Expectation),delimiter=',')

# np.savetxt('data/demonator.csv',pdf-np.square(pdf),delimiter=',')

#k_max = np.argmax(square_err)

plt.figure("square_err")
plt.plot(square_err)

apart_value = 160

square_err = square_err[apart_value:]

tmp_min = np.min(square_err[square_err!=0])

k_max = np.array(np.where(square_err==tmp_min))[0][0]+apart_value

# np.savetxt('data/square_err.csv',square_err,delimiter=',')

# print(k_max)

# interval = 10

# data = square_err[k_max-interval:k_max+interval:1]

# w1 = np.random.normal(loc=k_max, scale=1, size=data.size)

# k_max = int(np.average(data, axis=0, weights=w1))

print("kmax = ",k_max)


#200 as tmp threshold

ret4,th4 = cv.threshold(gray,100,255,cv.THRESH_BINARY)



plt.figure('main component2')





th4_tmp = cv.merge((th4,th4,th4))
res = cv.bitwise_and(test_plt,th4_tmp)
res = np.vstack((test_plt,th4_tmp,res))

plt.imshow(res)





# kernel = np.array([[0,0,0,1,0,0,0],[0,0,1,1,1,0,0],[0,1,1,1,1,1,0],[0,0,1,1,1,0,0],[0,0,0,1,0,0,0]],dtype=np.uint8)

kernel1 = np.ones((3,3),np.uint8)




opening = cv.morphologyEx(th4,cv.MORPH_OPEN,kernel1)

plt.figure('open')
plt.imshow(th4)

contours,hierarchy = cv.findContours(th4, 1, 2)

print("num of contours",len(contours))


contour_img = cv.drawContours(test, contours, -1, (0,255,255), 3)


cxy = np.zeros((11,2))
for i,cnt in enumerate(contours):

    M = cv.moments(cnt)
    cxy[i][0] = int(M['m10']/M['m00'])
    cxy[i][1] = int(M['m01']/M['m00'])

point_size = 1
point_color = (0, 0, 255) # BGR
thickness = -1 #  0 4 8
cxy = cxy.astype(np.uint32)
#dot picked

# np.savetxt('data/dot_picked_raw.csv',cxy,delimiter=',')


for pointx,pointy in cxy:
	cv.circle(test, (pointx,pointy), point_size, point_color, thickness)
    
tmp_cxy = cxy[5:]

# print(tmp_cxy)

tmp_order = np.array([5,6,4,2,1,3])
tmp_order = tmp_order - 1

print(tmp_cxy[tmp_order])

plt.figure('polylines_img')

linespt = tmp_cxy[tmp_order].astype(np.int32)

polylines_img = cv.fillPoly(test,[linespt],(255,0,0))



plt.imshow(polylines_img)






plt.figure('contour_img')

plt.imshow(contour_img)

plt.figure('picked out')

test_plt = test[:,:,::-1]

plt.imshow(test_plt)

plt.show()



