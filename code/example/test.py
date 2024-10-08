#encoding=utf-8
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sg

def get_maxima(values: np.ndarray):

    max_index = sg.argrelmax(values)[0]
    return max_index, values[max_index]
#read test img

test = cv.imread('pic/3.jpg')


#cvtgray

gray = cv.cvtColor(test,cv.COLOR_BGR2GRAY)

zero_like_mat = np.zeros_like(gray)

#calc hist         

hist = cv.calcHist([gray],[0],None,[256],[0,256])

max_index,values = get_maxima(hist)

index = np.argsort(values,axis=0)

flag = 1

tmp = index[flag][0]

designed_method = gray[gray > tmp]

hist_new = cv.calcHist([designed_method],[0],None,[256],[0,256])

#calc pdf

pdf = hist_new / designed_method.shape

cdf = pdf.cumsum()

sequence = np.arange(0,256)

pdf = pdf[:,0]

Expectation = pdf*sequence

eps = np.finfo(Expectation.dtype).eps

eps_sequence =  eps*np.ones(256)

square_err = np.divide(np.square(Expectation[255]*pdf-Expectation),(pdf-np.square(pdf)+eps))

apart_value = 160

square_err = square_err[apart_value:]

tmp_min = np.min(square_err[square_err!=0])

k_max = np.array(np.where(square_err==tmp_min))[0][0]+apart_value

print("kmax = ",k_max)

ret4,th4 = cv.threshold(gray,100,255,cv.THRESH_BINARY)

plt.figure('threshold')

plt.imshow(th4)

plt.show()




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
print('cxy=',cxy)
# np.savetxt('data/dot_picked_raw.csv',cxy,delimiter=',')


for pointx,pointy in cxy:
	cv.circle(test, (pointx,pointy), point_size, point_color, thickness)
    
tmp_cxy = cxy

# print(tmp_cxy)

# tmp_order = np.array([5,6,4,2,1,3])
# tmp_order = tmp_order - 1

# plt.figure('contour_img')

# plt.imshow(contour_img)

# plt.show()

dots = tmp_cxy[:6,:].astype(np.int32)

dots = np.array(dots)


mid_dot_x = np.average(dots[:,0])
mid_dot_y = np.average(dots[:,1])

mid_dot = np.array([mid_dot_x,mid_dot_y])
# print(mid_dot)

initial = 0

dot = dots[initial,:]

angle = np.zeros(6)

def calc_angle(tmp):

    u = tmp-mid_dot
    v = dot-mid_dot
    cos_ = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    sin_ = np.cross(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    angle = np.arctan2(sin_,cos_)
    return angle

for i,tmp in enumerate(dots):
    if i == initial:
        continue
    angle[i] = calc_angle(tmp)
print(angle/3.1415*180)

sort_index = np.argsort(angle)

print(sort_index)






# plt.figure('dots pic')

# plt.scatter(dots[:,0],dots[:,1],)

# plt.scatter(mid_dot_x,mid_dot_y)

# ax = plt.gca()

# ax.invert_yaxis() 

# plt.show()


# test pass

# dots = dots[sort_index] 
# blank = np.zeros((450,800))
# test = polylines_img = cv.polylines(blank,[dots],True,(255,255,255))
# cv.imshow('test',test)

# cv.waitKey(0)

# cv.destroyAllWindows()

#next distinguish two character to make sure the direction



dots = dots[sort_index]

print('dots = ',dots)

vector_angle = np.zeros(6)

len_dots = len(dots)



for i in np.arange(len_dots):
    if i ==0:
        vector_angle[i] = np.dot(dots[-1]-dots[i],dots[i+1]-dots[i])/(np.linalg.norm(dots[-1]-dots[i]) * np.linalg.norm(dots[i+1]-dots[i]))
    elif i==len_dots-1:
        vector_angle[i] = np.dot(dots[i-1]-dots[i],dots[0]-dots[i])/(np.linalg.norm(dots[i-1]-dots[i]) * np.linalg.norm(dots[0]-dots[i]))
    else:
        vector_angle[i] = np.dot(dots[i-1]-dots[i],dots[i+1]-dots[i])/(np.linalg.norm(dots[i-1]-dots[i]) * np.linalg.norm(dots[i+1]-dots[i]))

vector_angle=np.arccos(vector_angle)/np.pi*180

print(vector_angle)

## look for the 130-150 and the 170

index5 = 0

index2 = 0

for i,tmp in enumerate(vector_angle):

    if tmp>170:
        index5 = i
    elif tmp >120 and tmp <150:
        index2  = i
    else:
        continue

print('index2==',index2,'index5==',index5)

offset  = 2-index2

dots = np.roll(dots,offset,axis = 0)

for i in np.arange(6):

    cv.putText(contour_img,'{}'.format(i),(dots[i,:]),cv.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255),2,lineType=cv.LINE_AA)

#一个条件就够，俩条件可以互相验证 2 -index2 = 5 -index5 = 右移单位
cv.imshow('res',contour_img)
cv.waitKey(0)
cv.destroyAllWindows()

# plt.figure('picked out')

# test_plt = test[:,:,::-1]

# plt.imshow(test_plt)

# plt.show()











# questionCnts = []
for c in contours:
	# 计算比例和大小
	(x, y, w, h) = cv.boundingRect(c)
	ar = w / float(h)

	# 根据实际情况指定标准
	if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
		questionCnts.append(c)

# # 按照从上到下进行排序