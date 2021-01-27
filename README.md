# Détecteur des masques

## Baptiste LB & Thomas

---

### Etapes pour l'entrainement du modèle

[x] Transformer les images en matrices   
[x] Séparer les jeux d'entrainement et de test en deux matrices distinctes    
[x] Créer modèle et l'entrainer sur le jeu d'entrainement    
[x] Evaluer le modèle sur le jeu de test    
[x] Corriger le modèle jusqu'à obtenir des résultats concluants    

### Etapes pour le programme de détection en temps réel

[x] Importer le modèle dans le programme    
[x] Ecrire un programme qui lit le flux video de la webcam    
[x] Appliquer la prediction du modèle sur le flux vidéo    

### Rédaction du rapport 

[x] Rédiger le rapport   

### Lien pour obtenir le model

[Lien WeTransfer](https://we.tl/t-4SvTrIAePc)

### Comment tester l'application vidéo

Pour tester l'application de la détection du masque en temps réel :
- Télécharger le modèle sur le lien ci-dessus
- Executer le fichier `real_time.py` dans le `cmd`, à l'aide de la commande suivante : `python3 real_time.py`
- Pour fermer l'application, il faut fermer le `cmd` ou appuyer sur la touche `ESC`

### Pré-requis

- `pip install pydub`
- installer [ffmpeg](https://ffmpeg.org/download.html)
