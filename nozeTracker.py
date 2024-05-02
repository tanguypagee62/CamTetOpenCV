import cv2

def main():
    # Charger le modèle de détection de visages pré-entraîné
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Ouvrir la webcam
    cap = cv2.VideoCapture(0)

    # Créer une nouvelle fenêtre avec une taille spécifique
    cv2.namedWindow('Focus sur le nez', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Focus sur le nez', 800, 600)  # Ajuster la taille selon vos besoins

    while True:
        # Lire une trame de la webcam
        ret, frame = cap.read()

        # Convertir l'image en niveaux de gris pour la détection de visages
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Détecter les visages dans l'image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Pour chaque visage détecté
        for (x, y, w, h) in faces:
            # Calculer la position du nez
            nose_x = x + w // 2
            nose_y = y + h // 2

            # Dessiner un rectangle autour du visage
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Dessiner un cercle sur le nez
            cv2.circle(frame, (nose_x, nose_y), 5, (0, 255, 0), -1)

            # Zoom sur le nez (ajuster la taille de la région zoomée selon vos besoins)
            zoom_factor = 2
            zoomed_frame = frame[max(0, nose_y - 100*zoom_factor):min(frame.shape[0], nose_y + 100*zoom_factor),
                                 max(0, nose_x - 100*zoom_factor):min(frame.shape[1], nose_x + 100*zoom_factor)]

            # Afficher la trame zoomée dans la nouvelle fenêtre
            cv2.imshow('Focus sur le nez', zoomed_frame)

        # Attendre 1 milliseconde et vérifier si l'utilisateur appuie sur la touche 'q' pour quitter
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libérer la webcam et fermer la fenêtre d'affichage
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
