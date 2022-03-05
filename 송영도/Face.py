#!/usr/bin/env python
# coding: utf-8

# In[1]:


from statistics import mode

import cv2
from keras.models import load_model
import numpy as np
import utils
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input
#만약 프롬포트에서 python으로 import tensorflow가 되고 주피터 커널로 실행이 안된다면 pip install ipykernel해서 커널을 설치하고 다시 연결
#pip install tensorflow가 안된다면 지우고 다시해봄 그리고 파이썬을 지웠다가 아나콘다 파이썬 버전과 맞춰줌 pip install h5py와 같이
#안됐던 이유는 아마 텐서플로의 버전과 파이썬 버전때문이 아닌듯 의심, python해서 import tensorflow가 된다면 커널문제이니 재설치등
#h5py\h5.pyx in init h5py.h5() 
#**AttributeError: type object 'h5py.h5.H5PYConfig' has no attribute '__reduce_cython__'**
#오류의 경우 지우고 다시 깔거나 pip install cython 그리고 파이썬에서 바로 확인


# In[2]:


import os 
print(os.getcwd())


# In[3]:


import sys
print(sys.version)


# In[4]:


import tensorflow as tf
tf.__version__


# In[61]:


# parameters for loading data and images
detection_model_path = 'trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = 'trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
emotion_labels = get_labels('fer2013')
net = cv2.dnn.readNetFromTorch('models/instance_norm/starry_night.t7')#화풍변경
# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

# loading models
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# starting lists for calculating modes
emotion_window = []


# In[14]:


# # video streaming
# cv2.namedWindow('window_frame')
# video_capture = cv2.VideoCapture(0)

# while True:
#     bgr_image = video_capture.read()[1]
#     gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
#     rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
#     faces = detect_faces(face_detection, gray_image)

#     for face_coordinates in faces:

#         x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
#         gray_face = gray_image[y1:y2, x1:x2]
#         try:
#             gray_face = cv2.resize(gray_face, (emotion_target_size))
#         except:
#             continue

#         gray_face = preprocess_input(gray_face, True)
#         gray_face = np.expand_dims(gray_face, 0)
#         gray_face = np.expand_dims(gray_face, -1)
#         emotion_prediction = emotion_classifier.predict(gray_face)
#         emotion_probability = np.max(emotion_prediction)
#         emotion_label_arg = np.argmax(emotion_prediction)
#         emotion_text = emotion_labels[emotion_label_arg]
#         emotion_window.append(emotion_text)

#         if len(emotion_window) > frame_window:
#             emotion_window.pop(0)
#         try:
#             emotion_mode = mode(emotion_window)
#         except:
#             continue

#         if emotion_text == 'angry':
#             color = emotion_probability * np.asarray((255, 0, 0))
#         elif emotion_text == 'sad':
#             color = emotion_probability * np.asarray((0, 0, 255))
#         elif emotion_text == 'happy':
#             color = emotion_probability * np.asarray((255, 255, 0))
#         elif emotion_text == 'surprise':
#             color = emotion_probability * np.asarray((0, 255, 255))
#         else:
#             color = emotion_probability * np.asarray((0, 255, 0))

#         color = color.astype(int)
#         color = color.tolist()

#         draw_bounding_box(face_coordinates, rgb_image, color)
#         draw_text(face_coordinates, rgb_image, emotion_mode,
#                   color, 0, -45,2, 5)

#     bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    
    
#     MEAN_VALUE = [103.939, 116.779, 123.680]
#     blob = cv2.dnn.blobFromImage(bgr_image, mean=MEAN_VALUE)

#     net.setInput(blob)
#     output = net.forward()

#     output = output.squeeze().transpose((1, 2, 0))

#     output += MEAN_VALUE
#     output = np.clip(output, 0, 255)
#     output = output.astype('uint8')
    
#     cv2.imshow('window_frame', output)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# video_capture.release()
# cv2.destroyAllWindows()


# In[62]:


# By image
img = cv2.imread('images/03.jpg')
img=cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE)#90도 반시계
img=cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)#90도 시계
bgr_image = img
gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
faces = detect_faces(face_detection, gray_image)

for face_coordinates in faces:

    x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
    gray_face = gray_image[y1:y2, x1:x2]
    try:
        gray_face = cv2.resize(gray_face, (emotion_target_size))
    except:
        continue

    gray_face = preprocess_input(gray_face, True)
    gray_face = np.expand_dims(gray_face, 0)
    gray_face = np.expand_dims(gray_face, -1)
    emotion_prediction = emotion_classifier.predict(gray_face)
    emotion_probability = np.max(emotion_prediction)
    emotion_label_arg = np.argmax(emotion_prediction)
    emotion_text = emotion_labels[emotion_label_arg]
    emotion_window.append(emotion_text)

    if len(emotion_window) > frame_window:
        emotion_window.pop(0)
    try:
        emotion_mode = mode(emotion_window)
    except:
        continue

    if emotion_text == 'angry':
        color = emotion_probability * np.asarray((255, 0, 0))
    elif emotion_text == 'sad':
        color = emotion_probability * np.asarray((0, 0, 255))
    elif emotion_text == 'happy':
        color = emotion_probability * np.asarray((255, 255, 0))
    elif emotion_text == 'surprise':
        color = emotion_probability * np.asarray((0, 255, 255))
    else:
        color = emotion_probability * np.asarray((0, 255, 0))

    color = color.astype(int)
    color = color.tolist()

    draw_bounding_box(face_coordinates, rgb_image, color)
    draw_text(face_coordinates, rgb_image, emotion_mode,
              color, 0, -45, 2, 5)

bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

MEAN_VALUE = [103.939, 116.779, 123.680]
blob = cv2.dnn.blobFromImage(bgr_image, mean=MEAN_VALUE)

net.setInput(blob)
output = net.forward()

output = output.squeeze().transpose((1, 2, 0))

output += MEAN_VALUE
output = np.clip(output, 0, 255)
output = output.astype('uint8')




cv2.imshow('before', cv2.resize(img,(540,720)))
cv2.imshow('window_frame', cv2.resize(output,(540,720)))

cv2.waitKey(0)
cv2.destroyAllWindows()

