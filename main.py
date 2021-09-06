import cv2
import mediapipe as mp
import time


cap = cv2.VideoCapture(0)

mpHand = mp.solutions.hands
hands = mpHand.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    succ, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handsLm in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handsLm,mpHand.HAND_CONNECTIONS)

    cv2.imshow("Image", img)
    cv2.waitKey(1)


