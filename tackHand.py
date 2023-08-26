import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils  # mediapipe drawing Utils
previous_time = 0
current_time = 0

while cap.isOpened:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)


    if results.multi_hand_landmarks:
        for hand_land_marks in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, hand_land_marks, mpHands.HAND_CONNECTIONS)

    current_time = time.time()
    fps = 1/(current_time-previous_time)
    previous_time = current_time

    cv2.putText(img, str(int(fps)), (10,70),cv2.FONT_HERSHEY_COMPLEX,3,(255,180,120),3)

    cv2.imshow('image', img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
   