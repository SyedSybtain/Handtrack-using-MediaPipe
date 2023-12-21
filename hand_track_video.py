import cv2
import mediapipe as mp


vid = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
Hands = mpHands.Hands()

while True:
	success,img = vid.read()
	imgRGB = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
	result = Hands.process(imgRGB)

	if result.multi_hand_landmarks:
		for multMrk in result.multi_hand_landmarks:
			mpDraw.draw_landmarks(img,multMrk , mpHands.HAND_CONNECTIONS)
	cv2.imshow('img',img)
	cv2.waitKey(0)
	# cv2.destroyAllWindows()