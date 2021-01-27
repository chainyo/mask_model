import tensorflow as tf 
import cv2 

from pydub import AudioSegment
from pydub.playback import play

model = tf.keras.models.load_model("mask_model")

video = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('files/haarcascade_frontalface_default.xml')
cnt = 0

while(True):

    ret, img = video.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    song = AudioSegment.from_mp3('./files/beep-07.mp3')
    song = song - 40

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        pred = cv2.resize(gray, (100,100))
        pred = pred.reshape(-1, 100, 100, 1)
        y_pred = model.predict_classes(pred)

        if y_pred[0] == 1:
            cv2.putText(img, 'FELICITATIONS POUR LE PORT DU MASQUE', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (49,193,20), 2)
        else:
            cv2.putText(img, 'MERCI DE METTRE UN MASQUE', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (39,11,240), 2)
            if cnt == 0:
                play(song)
                cnt += 1
            elif cnt == 10:
                cnt = 0
            else:
                cnt += 1

    cv2.imshow('img', img)

    k = cv2.waitKey(30) & 0xff
    if k==27:
        break


video.release()
cv2.destroyAllWindows()