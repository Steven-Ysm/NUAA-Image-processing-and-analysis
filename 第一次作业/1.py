
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def imghistogram(img):
    """
    根据图像绘制直方图
    """
    h, w = img.shape  # 获取图像的高与宽
    hist = [0] * 256  # 首先各灰度频数都置为0
    for i in range(h):
        for j in range(w):
            hist[img[i, j]] += 1 #矩阵操作
    return hist
 
 
img = cv.imread("1.png", 0)  # 以灰度图像读入

plt.bar(range(256), imghistogram(img), width=1)
plt.grid(True,linestyle=':',color='r',alpha=0.6)
plt.title("Gray histogram")
plt.show()

 
import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('4.png',0)
img_copytwo = cv2.imread('4.png',0)
img_copythree = cv2.imread('4.png',0)
img_copyfour = cv2.imread('4.png',0)
img_copyfive = cv2.imread('4.png',0)
img_copysix = cv2.imread('4.png',0)

h = img.shape[0]
w = img.shape[1]


def median_filter(x, y, step):
    sum_s=[] # 定义空数组
    for k in range(-int(step/2), int(step/2)+1):
        for m in range(-int(step/2), int(step/2)+1):
            sum_s.append(img[x+k][y+m]) # 把模块的像素添加到空数组
    sum_s.sort() # 对模板的像素由小到大进行排序
    return sum_s[(int(step*step/2)+1)]

MedStep = [2,3,4,5,6] #设置滤波器

for k in range(0,5):
    medstep = MedStep[k]
    for i in range(int(medstep/2) ,h - int(medstep/2)):
        for j in range(int(medstep/2) ,w - int(medstep/2)):
            if(medstep == 2):
                img_copytwo[i][j] = median_filter(i, j, medstep) 
            elif(medstep == 3):
                img_copythree[i][j] = median_filter(i, j, medstep) 
            elif(medstep == 4):
                img_copyfour[i][j] = median_filter(i, j, medstep) 
            elif(medstep == 5):
                img_copyfive[i][j] = median_filter(i, j, medstep) 
            elif(medstep == 6):
                img_copysix[i][j] = median_filter(i, j, medstep) 

#创建一个窗口
plt.figure('contrast',figsize=(7,5))
#显示原图
plt.subplot(321) #子图1
plt.imshow(img,plt.cm.gray)
plt.title('before')
 
#显示处理过的图像

plt.subplot(322) #子图2
plt.imshow(img_copytwo,plt.cm.gray)
plt.title('filter_two')

plt.subplot(323) #子图3
plt.imshow(img_copythree,plt.cm.gray)
plt.title('filter_three')

plt.subplot(324) #子图4
plt.imshow(img_copyfour,plt.cm.gray)
plt.title('filter_four')

plt.subplot(325) #子图5
plt.imshow(img_copyfive,plt.cm.gray)
plt.title('filter_five')

plt.subplot(326) #子图6
plt.imshow(img_copysix,plt.cm.gray)
plt.title('filter_six')

plt.savefig('1.jpg')
plt.show()


import numpy as np
import cv2 as cv

img = cv.imread("2.png")		
B,G,R = cv.split(img)
BH = cv.equalizeHist(B) #分别对3通道进行直方图均衡化
GH = cv.equalizeHist(G)
RH = cv.equalizeHist(R)
result = cv.merge((BH,GH,RH),)#通道合成
res = np.hstack((img,result))

cv.imwrite("after2.png", res)

import numpy as np
import cv2 as cv

img = cv.imread("3.png")		
B,G,R = cv.split(img)
BH = cv.equalizeHist(B) #分别对3通道进行直方图均衡化
GH = cv.equalizeHist(G)
RH = cv.equalizeHist(R)
result = cv.merge((BH,GH,RH),)#通道合成
res = np.hstack((img,result))

cv.imwrite("after3.png", res)