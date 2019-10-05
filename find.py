import cv2
import numpy as np
import matplotlib.pyplot as plt
# 1 par
PosNum = 820
NegNum = 1931
winSize = (64,128)
blockSize = (16,16)# 105
blockStride = (8,8)#4 cell
cellSize = (8,8)
nBin = 9#9 bin 3780

hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nBin)
# 3 svm
svm = cv2.ml.SVM_create()
# 4 computer hog
featureNum = int(((128-16)/8+1)*((64-16)/8+1)*4*9) #3780
featureArray = np.zeros(((PosNum+NegNum),featureNum),np.float32)
labelArray = np.zeros(((PosNum+NegNum),1),np.int32)

for i in range(0,PosNum):
    fileName = 'pos/'+str(i+1)+'.jpg'
    img = cv2.imread(fileName)
    hist = hog.compute(img,(8,8))# 3780
    for j in range(0,featureNum):
        featureArray[i,j] = hist[j]
    # featureArray hog [1,:] hog1 [2,:]hog2 
    labelArray[i,0] = 1
    # 正样本 label 1
    
for i in range(0,NegNum):
    fileName = 'neg/'+str(i+1)+'.jpg'
    img = cv2.imread(fileName)
    hist = hog.compute(img,(8,8))# 3780
    for j in range(0,featureNum):
        featureArray[i+PosNum,j] = hist[j]
    labelArray[i+PosNum,0] = -1
# 负样本 label -1
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setC(0.01)

# 6 train
svm.train(featureArray,cv2.ml.ROW_SAMPLE,labelArray)
resultArray = np.zeros((1,featureNum),np.float32)
# detect
myDetect = np.zeros((3780),np.float32)
for i in range(0,3780):
    myDetect[i] = 0
myHog = cv2.HOGDescriptor()
myHog.setSVMDetector(myDetect)

# load 
imageSrc = cv2.imread('Test2.jpg',1)
# (8,8) win 
objs = myHog.detectMultiScale(imageSrc,0,(8,8),(32,32),1.05,2)
# xy wh 三维 最后一维
x = int(objs[0][0][0])
y = int(objs[0][0][1])
w = int(objs[0][0][2])
h = int(objs[0][0][3])
# 绘制展示


cv2.rectangle(imageSrc,(x,y),(x+w,y+h),(255,0,0),2)
cv2.imshow('dst',imageSrc)
cv2.waitKey(0)
