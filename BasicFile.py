import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0


while True:
    success, image = cap.read()

    imgRGB = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                #print(id,lm)
                h,w,c = image.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                #id number, cx, cy position
                print(id, cx, cy)
                #if id == 0:
                cv.circle(image, (cx,cy), 3, (0,220,0), cv.FILLED)
            mpDraw.draw_landmarks(image, handLms, mpHands.HAND_CONNECTIONS)



    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv.putText(image, str(int(fps)), (10,70), cv.FONT_HERSHEY_PLAIN, 2, (255,255,255), 3)

    cv.imshow("Image", image)
    key = cv.waitKey(3000)
    if key & 0xFF == ord('d'):
        break

cap.release()
cv.destroyAllWindows()


