import cv2
import numpy as np
#引入模型库
face_xml = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_xml = cv2.CascadeClassifier('haarcascade_eye.xml')
# load jpg
img = cv2.imread('girl.jpg')
# 灰化图像
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# detect faces 1 data 2 scale 3 5
faces = face_xml.detectMultiScale(gray,1.3,5)
print('face=',len(faces))
# draw
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_face = gray[y:y+h,x:x+w]
    # 1 gray
    eyes = eye_xml.detectMultiScale(roi_face)
    print('eye=',len(eyes))
    #for (e_x,e_y,e_w,e_h) in eyes:
        #cv2.rectangle(roi_color,(e_x,e_y),(e_x+e_w,e_y+e_h),(0,255,0),2)
cv2.imshow('abcd',img)
cv2.waitKey(0)
