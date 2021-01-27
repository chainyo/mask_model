import tensorflow as tf 
import cv2 

from pydub import AudioSegment
from pydub.playback import play

# Import du modèle
model = tf.keras.models.load_model("mask_model")

# Instance de l'outil de capture vidéo
video = cv2.VideoCapture(0)

# Import du modèle de reconnaisssance faciale
face_cascade = cv2.CascadeClassifier('files/haarcascade_frontalface_default.xml')

# Compteur pour la musique
cnt = 0

# Boucle pour la lecture du flux vidéo
while(True):

    # On récupère les infos du flux vidéo pour pouvoir les utiliser
    ret, img = video.read()
    # On convertit le flux en gris pour notre modèle
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # import de la détection de visage
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    # import du song et réduction du niveau sonore
    song = AudioSegment.from_mp3('./files/beep-07.mp3')
    song = song - 40

    # Pour l'affichage des carrés de détection du visage
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Prédiction à envoyer à notre modèle
        pred = cv2.resize(gray, (100,100))
        pred = pred.reshape(-1, 100, 100, 1)

        # On fait la prédiction sur l'image envoyée à notre modèle
        y_pred = model.predict_classes(pred)

        # En fonction du résultat des prédictions, on affiche un message et on joue un son
        if y_pred[0] == 1:
            cv2.putText(img, 'FELICITATIONS POUR LE PORT DU MASQUE', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (49,193,20), 2)
        else:
            cv2.putText(img, 'MERCI DE METTRE UN MASQUE', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (39,11,240), 2)
            # Gestion du compteur pour jouer le son mais pas en permanence
            if cnt == 0:
                play(song)
                cnt += 1
            elif cnt == 10:
                cnt = 0
            else:
                cnt += 1

    cv2.imshow('img', img)

    # On définit une touche pour fermer l'application
    k = cv2.waitKey(30) & 0xff
    if k==27: # touche 'ESC' pour fermer l'application
        break

# On detruit l'objet vidéo instancié et on ferme toutes les fenêtres
video.release()
cv2.destroyAllWindows()