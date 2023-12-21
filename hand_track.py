import cv2
import mediapipe as mp
import time

img = cv2.imread("hand2.jpg")

imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

results = hands.process(imgRGB)

# print(results.multi_hand_landmarks)

if results.multi_hand_landmarks:
	for HandLms in results.multi_hand_landmarks:
		mpDraw.draw_landmarks(img, HandLms , mpHands.HAND_CONNECTIONS)

cv2.imshow("My Image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()