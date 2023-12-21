import cv2
import mediapipe as mp

img = cv2.imread("hand2.jpg")
imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

mpHands = mp.solutions.hands
Hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

results = Hands.process(imgRGB)

if results.multi_hand_landmarks:
	for handLms in results.multi_hand_landmarks:
		for i_d,lm in enumerate(handLms.landmark):
			# print(i_d)
			h,w,c = img.shape

			cx,cy = int(lm.x*w),int(lm.y*h)

			if i_d == 12:
				cv2.circle(img,(cx,cy),25,(200,200,0),cv2.FILLED)
				print("The thumb is at pixel vales : ",cx,cy)

		mpDraw.draw_landmarks(img,handLms, mpHands.HAND_CONNECTIONS)
		# mpDraw.draw_landmarks(img,handLms)


cv2.imshow("My Image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()




