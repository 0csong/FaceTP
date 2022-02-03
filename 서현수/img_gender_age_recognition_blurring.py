#!/usr/bin/env python
# coding: utf-8

# In[1]:


from imutils import face_utils
import numpy as np
import cv2
import glob
import dlib

ALL = list(range(0, 68))
age_list = ['(0-2)','(4-6)','(8-12)','(15-20)','(25-32)','(38-43)','(48-53)','(60-100)']
gender_list = ['Male', 'Female']

predictor_file = 'shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_file)
detector = dlib.get_frontal_face_detector()
age_net = cv2.dnn.readNetFromCaffe(
          'age_deploy.prototxt', 
          'age_net.caffemodel')
gender_net = cv2.dnn.readNetFromCaffe(
          'gender_deploy.prototxt', 
          'gender_net.caffemodel')

img_file = glob.glob('dami.jpg')

for img_path in img_file:
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    print("Number of faces detected: {}".format(len(faces)))

                     
for (i, rect) in enumerate(faces):
    points = np.matrix([[p.x, p.y] for p in predictor(gray, rect).parts()])
    show_parts = points[ALL]  
    
    for (i, point) in enumerate(show_parts):
        x = point[0,0]
        y = point[0,1]
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        face_img = img[y:y + h, x:x + w].copy()
        blob = cv2.dnn.blobFromImage(face_img, scalefactor=1, size=(227, 227),
        mean=(78.4263377603, 87.7689143744, 114.895847746),
        swapRB=False, crop=False)
        
    # predict gender
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = gender_list[gender_preds[0].argmax()]

    # predict age
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = age_list[age_preds[0].argmax()]

    # visualize
    cv2.circle(img, (x, y), 1, (0, 255, 255), -1)
    #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    face_blur = cv2.GaussianBlur(face_img,(99,99), 30)
    img[y:y + h, x:x + w] = face_blur                             
    overlay_text = '%s %s' % (gender, age)
    cv2.putText(img, overlay_text, org=(x, y),
    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.45, color=(0,255,0), thickness=2)

cv2.imshow('img', img)
cv2.imwrite('result/%s' % img_path.split('/')[-1], img)
cv2.waitKey(0)  


# In[ ]:




