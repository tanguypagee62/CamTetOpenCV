import cv2
import os

def capture_photo(face_image):
    # Nom du répertoire où les photos seront sauvegardées
    output_dir = "img"

    # Créer le répertoire s'il n'existe pas
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Nom du fichier image
    filename = os.path.join(output_dir, "face_{}.jpg".format(len(os.listdir(output_dir))))

    # Sauvegarder l'image
    cv2.imwrite(filename, face_image)
    print("Photo enregistrée sous:", filename)

def main():
    # Charger le modèle de détection de visages pré-entraîné
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Ouvrir la webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Lire une trame de la webcam
        ret, frame = cap.read()

        # Convertir l'image en niveaux de gris pour la détection de visages
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Détecter les visages dans l'image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Pour chaque visage détecté
        for (x, y, w, h) in faces:
            # Dessiner un rectangle autour du visage
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Capturer une photo lorsque le visage est détecté
            capture_photo(frame[y:y+h, x:x+w])

        # Afficher le cadre avec les visages détectés
        cv2.imshow('Capture photo discrète', frame)

        # Attendre 1 milliseconde et vérifier si l'utilisateur appuie sur la touche 'q' pour quitter
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libérer la webcam et fermer la fenêtre d'affichage
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
