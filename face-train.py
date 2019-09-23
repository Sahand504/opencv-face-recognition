import os
import cv2
import numpy as np
import pickle
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, 'images')
FACE_DIR = os.path.join(IMAGE_DIR, 'faces')

x_train = []
y_labels = []

current_id = 0
label_ids = {}

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

for root, dirs, files in os.walk(FACE_DIR):
    for file in files:
        if file.endswith('png') or file.endswith('PNG') \
                or file.endswith('jpg') or file.endswith('JPG') or \
                file.endswith('jpeg') or file.endswith('JPEG'):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace('_', ' ').lower()

            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
                _id = label_ids[label]

            pil_image = Image.open(path).convert('L')  # Grayscale
            gray_img = np.array(pil_image, 'uint8')

            detected_faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.5, minNeighbors=5)

            for (x, y, w, h) in detected_faces:
                gray_face_img = gray_img[y:y + h, x:x + w]  # region of interest
                x_train.append(gray_face_img)
                y_labels.append(_id)
                print(label)
                print(path)
                print(_id)

with open('labels.pickle', 'wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save('trainner.yml')
