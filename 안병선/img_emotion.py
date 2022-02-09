import cv2
import numpy as np
import glob
import sys
from keras.preprocessing.image import img_to_array
from keras.models import load_model

img_files = glob.glob('.\img\*.jpg')

# 이미지 없을때 예외처리
if not img_files:
    print("jpg 이미지가 없어요..")
    sys.exit()

# Face detection XML load and trained model loading
face_detection = cv2.CascadeClassifier('files/haarcascade_frontalface_default.xml')
emotion_classifier = load_model('files/emotion_model.hdf5', compile=False)

# hdf5 대용양 파일 사용 위한 형식, XML과 동일하다. 속도가 빠르고 안정적이다.
EMOTIONS = ["Angry", "Disgusting", "Fearful", "Happy", "Sad", "Surpring", "Neutral"]

for i in range(len(img_files)):
    img = cv2.imread(img_files[i])
    # Convert color to gray scale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Face detection in frame
    faces = face_detection.detectMultiScale(gray,
                                            scaleFactor=1.1,
                                            minNeighbors=5,
                                            minSize=(30, 30))
    # Create empty image
    canvas = np.zeros((250, 300, 3), dtype="uint8")
    print(faces)
    # Perform emotion recognition only when face is detected
    for i in range(len(faces)):
        # For the largest image
        face = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))
        (fX, fY, fW, fH) = face[i]
        # Resize the image to 48x48 for neural network
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # Emotion predict
        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]

        # Assign labeling
        cv2.putText(img, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(img, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)

        # Label printing
        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
            text = "{}: {:.2f}%".format(emotion, prob * 100)
            w = int(prob * 300)
            cv2.rectangle(canvas, (7, (i * 35) + 5), (w, (i * 35) + 35), (0, 0, 255), -1)
            cv2.putText(canvas, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

        # Open two windows
        ## Display image ("Emotion Recognition")
        ## Display probabilities of emotion
        cv2.imshow('Emotion Recognition', img)
        cv2.imshow("Probabilities", canvas)
        cv2.waitKey(0)
