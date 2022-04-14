import numpy as np
import cv2
import time
import pandas as pd

face_model = 'res10_300x300_ssd_iter_140000.caffemodel'  # dnn 딥러닝 방식을 이용한 face detection
face_prototxt = 'deploy.prototxt.txt'
age_model = 'age_net.caffemodel'
age_prototxt = 'age_deploy.prototxt'
gender_model = 'gender_net.caffemodel'
gender_prototxt = 'gender_deploy.prototxt'
# caffemodel 파일 : 얼굴 인식을 위해 ResNet 기본 네트워크를 사용하는
#                   SSD(Single Shot Detector) 프레임워크를 통해 사전 훈련된 모델 가중치 사용
# prototxt 파일 : 모델의 레이어 구성 및 속성 정의

age_list = ['(0-10)', '(11-20)', '(21-30)', '(31-40)', '(41-50)', '(51-60)', '(61-70)', '(71-80)']
gender_list = ['Male', 'Female']

age_index = [0, 0, 0, 0, 0, 0, 0, 0]
gender_index = [0, 0]

title_name = 'Age and Gender Recognition'
min_confidence = 0.5  # 인식할 최소 확률
recognition_count = 0
elapsed_time = 0
OUTPUT_SIZE = (300, 300)

detector = cv2.dnn.readNetFromCaffe(face_prototxt, face_model)
age_detector = cv2.dnn.readNetFromCaffe(age_prototxt, age_model)
gender_detector = cv2.dnn.readNetFromCaffe(gender_prototxt, gender_model)

def detectAndDisplay(image):
    start_time = time.time()
    (h, w) = image.shape[:2]

    roi_x, roi_y, roi_w, roi_h = 0, h // 2, w, 2
    cv2.rectangle(image, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 0, 255), 2)

    imageBlob = cv2.dnn.blobFromImage(image, 1.0, OUTPUT_SIZE,
                                      (104.0, 177.0, 123.0), swapRB=False, crop=False)
    # 1.0->scale, (104.0, 177.0, 123.0)-> min subtraction

    detector.setInput(imageBlob)
    detections = detector.forward()  # 얼굴 인식

    for i in range(0, detections.shape[2]):  # 여러명 얼굴 인식 가능
        confidence = detections[0, 0, i, 2]  # 얼굴 인식 확률 추출

        if confidence > min_confidence:  # 얼굴 인식 확률이 최소 확률보다 큰 경우
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]
            center_X = (startX + endX) / 2
            center_Y = (startY + endY) / 2


            face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                              (78.4263377603, 87.7689143744, 114.895847746),
                                              swapRB=False)  # min subtraction?

            age_detector.setInput(face_blob)
            age_predictions = age_detector.forward()
            age_index = age_predictions[0].argmax()
            age = age_list[age_index]
            age_confidence = age_predictions[0][age_index]

            gender_detector.setInput(face_blob)
            gender_predictions = gender_detector.forward()
            gender_index = gender_predictions[0].argmax()
            gender = gender_list[gender_index]
            gender_confidence = gender_predictions[0][gender_index]

            text = "{}: {:.2f}% {}: {:.2f}%".format(gender, gender_confidence * 100, age, age_confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            # cv2.rectangle(image, (startX, startY), (endX, endY),
            #    (0, 255, 0), 2)
            face_image = cv2.GaussianBlur(face, (99, 99), 30)  # 비식별화(가우시안 블러링)
            # cv2.GaussianBlur(입력 영상, 가우시안 커널의 크기, sigma)
            frame[startY:endY, startX:endX] = face_image
            cv2.putText(image, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
            print('==============================')
            print("Gender {} time {:.2f} %".format(gender, gender_confidence * 100))
            print("Age {} time {:.2f} %".format(age, age_confidence * 100))
            print("Age     Probability(%)")
            for i in range(len(age_list)):
                print("{}  {:.2f}%".format(age_list[i], age_predictions[0][i] * 100))
            print("Gender  Probability(%)")
            for i in range(len(gender_list)):
                print("{}  {:.2f} %".format(gender_list[i], gender_predictions[0][i] * 100))

            def age_num(num):
                global age_index
                if age == age_list[num]:
                    if ((roi_x < center_X < (roi_x + roi_w)) and (roi_y < center_Y < (roi_y + roi_h))):
                        age_index[num] += 1
                    else:
                        age_index[num] += 0
                return age_index[num]

            def gender_num(num):
                global gender_index
                if gender == gender_list[num]:
                    if ((roi_x < center_X < (roi_x + roi_w)) and (roi_y < center_Y < (roi_y + roi_h))):
                        gender_index[num] += 1
                    else:
                        gender_index[num] += 0
                return gender_index[num]

            age_values = [age_num(0), age_num(1), age_num(2), age_num(3), age_num(4), age_num(5), age_num(6), age_num(7)]
            gender_values = [gender_num(0), gender_num(1)]

            age_statistics = pd.DataFrame(age_list, columns=['Age'])
            age_statistics['Values'] = age_values
            print(age_statistics)
            gender_statistics = pd.DataFrame(gender_list, columns=['Gender'])
            gender_statistics['Values'] = gender_values
            print(gender_statistics)

            #face_blob.counted = True

    # frame_time = time.time() - start_time
    # global elapsed_time
    # elapsed_time += frame_time
    # print("Frame time {:.3f} seconds".format(frame_time)

    cv2.imshow(title_name, image)

filepath = 'yumi.mp4'
vs = cv2.VideoCapture(0)
time.sleep(2.0)
if not vs.isOpened:
    print('### Error opening video ###')
    exit(0)
while True:
    ret, frame = vs.read()
    if frame is None:
        print('### No more frame ###')
        vs.release()
        break
    detectAndDisplay(frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    if cv2.waitKey(1) == 27:
        break

vs.release()
cv2.destroyAllWindows()