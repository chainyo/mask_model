import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Fonction permettant de récupérer les images dans le dossier 
# Convertion en noir et blanc et redimensionnement en 100px/100px
def load_images_from_folder(folder):
    img = []
    for file in os.listdir(folder):
        i = cv2.imread(os.path.join(folder,file), cv2.IMREAD_GRAYSCALE)
        i = cv2.resize(i, (100,100))
        if i is not None:
            img.append(i)
    return img

# Train avec masque
images = load_images_from_folder('train/with_mask')
Train_with_mask = np.array(images)
# Train sans masque
images = load_images_from_folder('train/without_mask')
Train_without_mask = np.array(images)
# Test avec masque
images = load_images_from_folder('test/with_mask')
Test_with_mask = np.array(images)
# Test sans masque
images = load_images_from_folder('test/without_mask')
Test_without_mask = np.array(images)

# Création des matrices target de 1 et 0
y_train_mask = np.ones((len(Train_with_mask), 1))
y_train_no_mask = np.zeros((len(Train_without_mask), 1))

# Création des matrices target de 1 et 0
y_test_mask = np.ones((len(Test_with_mask), 1))
y_test_no_mask = np.zeros((len(Test_without_mask), 1))

# Jeu d'entraînement
y_train = np.concatenate((y_train_mask, y_train_no_mask))
X_train = np.concatenate((Train_with_mask, Train_without_mask))
# Jeu de test
y_test = np.concatenate((y_test_mask, y_test_no_mask))
X_test = np.concatenate((Test_with_mask, Test_without_mask))

# Mélange des jeux de données
array_random_train = np.random.permutation(len(y_train))
X_train = X_train[array_random_train]
y_train = y_train[array_random_train]
array_random_test = np.random.permutation(len(y_test))
X_test = X_test[array_random_test]
y_test = y_test[array_random_test]
