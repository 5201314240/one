import cv2
import numpy as np
from matplotlib import pyplot as plt

# 保存图片
image = cv2.imread('test.jpg')
cv2.imwrite('test.jpg',image)
# 图像拆分
b,g,r = cv2.split(image)
# 图像合并
image = cv2.merge((b,g,r))
# 图像属性
print(image.shape)
print(image.size)
print(image.dtype)
# 色彩转换
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# 直方图绘制
hist = cv2.calcHist([image],[0],None,[256],[0,256])
hist = plt.hist(image.ravel(),256,[0,256])
# 直方图正规化
image = cv2.normalize(image,image,255,0,cv2.NORM_MINMAX,cv2.CV_8U)
# 直方图均衡化
equ = cv2.equalizeHist(image)
# 自适应直方图均衡化
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
dst = clahe.apply(image)
# 高斯滤波
gauss = cv2.GaussianBlur(image,(5,5),0,0)
# 均值滤波
means = cv2.blur(image,(5,5))
# 中值滤波
median = cv2.medianBlur(image,5)
# 双边滤波
bilateral = cv2.bilateralFilter(image,5,5,5)
# 阈值分割
th,rst = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
# 图像侵蚀
k = np.ones((3,3),np.uint8)
img = cv2.erode(image,k,iterations=2)
# 图像膨胀
k = np.ones((3,3),np.uint8)
img1 = cv2.dilate(image,k,iterations=2)
# Sobel 边缘检测
Sobelx = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=-1)
Sobelx = cv2.convertScaleAbs(Sobelx)
Sobely = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=-1)
Sobely = cv2.convertScaleAbs(Sobely)
Sobelxy = cv2.addWeighted(Sobelx,0.5,Sobely,0.5,0)
# Canny 边缘检测
edg1 = cv2.Canny(image,100,200)
# 轮廓周长
image = cv2.imread("test.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # 转为灰度图
ret,binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY) # 二值化处理
contours ,hierarchy = cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
n = len(contours)
cntLen = []
for i in range(n):
    cntLen.append(cv2.arcLength(contours[i],True))
    print(f"第{i}个轮廓的长度是{cntLen[i]}")
# 轮廓面积
image = cv2.imread("test.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret,binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
contours,binary = cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
n = len(contours)
cntLen = []
for i in range(n):
    cntLen.append(cv2.contourArea(contours[i]))
    print(f'第{i}个轮廓的面积是{cntLen[i]}')

# 去云
image = cv2.imread('./process/2023-08-20-00_00_2023-08-20-23_59_Sentinel-2_L2A_True_color.jpg')
# 转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(gray)
_, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    # 使用形态学闭运算去除小的黑色区域（在二值图像中为云层）
    # 选择合适的核大小，这里假设为11x11
kernel = np.ones((11, 11), np.uint8)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
cloud_mask = cv2.bitwise_not(closing)
result = cv2.bitwise_and(image, image, mask=closing)
blur = cv2.blur(result,(5,5))
plt.imshow(blur[:,:,::-1])
blur = cv2.GaussianBlur(result,(3,3),1)
plt.imshow(blur[:,:,::-1])
blur = cv2.medianBlur(result,5)
plt.imshow(blur[:,:,::-1])
#