import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# [[379  77]
#  [420  74]
#  [414  93]
#  [423 111]
#  [381 114]
#  [380  95]]

dots = np.array([[702,81]
 ,[372,188]
 ,[502,303]
 ,[516,435]
 ,[800,342]
 [754,221]])
# dots = np.array(dots[:,0])


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

# print(sort_index)





plt.figure('dots pic')

plt.scatter(dots[:,0],dots[:,1])

plt.scatter(mid_dot_x,mid_dot_y)


plt.show()


# test pass

# dots = dots[sort_index] 
# blank = np.zeros((450,800))
# test = polylines_img = cv.polylines(blank,[dots],True,(255,255,255))
# cv.imshow('test',test)

# cv.waitKey(0)

# cv.destroyAllWindows()

#next distinguish two character to make sure the direction

print('dots',dots)

dots = dots[sort_index]

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

