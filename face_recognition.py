import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainner.yml')

labels = {"person_name": 1}
with open('labels.pickle', 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}

capture = cv2.VideoCapture(0)

while True:
    ret, img = capture.read()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detected_faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in detected_faces:
        gray_face_img = gray_img[y:y + h, x:x + w]  # region of interest
        color_face_img = img[y:y + h, x:x + w]
        _id, conf = recognizer.predict(gray_face_img)
        if 45 <= conf: #<= 85:
            print(labels[_id])
        cv2.imwrite("gray-face.png", gray_face_img)

        rec_color = (255, 0, 0)
        cv2.rectangle(img, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=2)

    cv2.imshow('Camera', img)
    if cv2.waitKey(1) == 27:
        break

capture.release()
cv2.destroyAllWindows()
