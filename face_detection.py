import cv2
import numpy as np

import keras
import tensorflow as tf

from keras.preprocessing.image import img_to_array

face_detector = cv2.CascadeClassifier('static/haarcascade_frontalface_default.xml')
classifier = keras.models.load_model('static/model.h5')
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


class DetectEmotion(object):

    def __init__(self):
        self.cap = cv2.VideoCapture(1)

    def __del__(self):
        self.cap.release()

    def get_frame(self):
        ret, frame = self.cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                preds = classifier.predict(roi)[0]
                label = class_labels[preds.argmax()]
                print('Face Detected! Emotion: ', label)

                label_position = (x, y)
                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

            else:
                cv2.putText(frame, 'No Face Detected!', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        ret, jpeg = cv2.imencode('.jpg', frame)

        return (jpeg.tobytes())