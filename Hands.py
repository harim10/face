import cv2
import mediapipe as mp

#Librerias y funciones principales de mediapipe
mp_drawing = mp.solutions.drawing_utils
#mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cam = cv2.VideoCapture(0)
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands = 2,
    min_detection_confidence=0.5) as hands:

    while True:
        ret, frame = cam.read()
        if ret == False:
            break

        #Tomamos la medida
        height, width, _ = frame.shape
        #Rotamos la imagen
        frame = cv2.flip(frame,1)
        #Transformamos de bgr a rgb
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks is not None:
        #Si tenemos informacion entonces con un form recorremos los datos y dibujamos
        #Esto es opcional y solo para personalizar la ubicación de los puntos checa documentación para puntos
            index = [5,9,13,17]
            for hand_landmarks in results.multi_hand_landmarks:
                #Dibujar los 21 puntos de deteccion
                mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                #Ciclo para mostar los puntos del index opcional
                for (i, points) in enumerate(hand_landmarks.landmark):
                    if i in index:
                        x = int(points.x * width)
                        y = int(points.y * height)
                        cv2.circle(frame, (x, y), 6, (0,0,255), -1)
                        cv2.circle(frame, (x, y), 3, (255,255,255), -1)
        
        #Mostramos

        frameHSV = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        cv2.imshow("Video normal",frame)
        
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

cam.realice()
cv2.destroyAllWindows()