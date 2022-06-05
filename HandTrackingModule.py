import cv2 as cv
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, maxHands=2, modelComplex=1,  detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.modelComplex = modelComplex

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, image, draw=True):

        imgRGB = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        #print(results.multi_hand_landmarks

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(image, handLms,
                                               self.mpHands.HAND_CONNECTIONS)

        return image

    def findPosition(self, image, handNu=0, draw = True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myhand = self.results.multi_hand_landmarks[handNu]
            for id, lm in enumerate(myhand.landmark):
                #print(id,lm)
                h, w, c = image.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                #id number, cx, cy position
                # print( id, cx, cy)
                lmList.append([id, cx, cy])
                if draw:
                #if id == 0:
                    cv.circle(image, (cx, cy), 10, (50, 205, 50), cv.FILLED)

        return lmList
    # if key & 0xFF == ord('d'):
    #     break

# cap.release()
# cv.destroyAllWindows()

def main():
    pTime = 0
    cTime = 0
    cap = cv.VideoCapture(0)
    detector = handDetector()

    while True:
        success, image = cap.read()
        image = detector.findHands(image)
        lmList = detector.findPosition(image)
        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv.putText(image, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)

        cv.imshow("Image", image)
        key = cv.waitKey(5000)





if __name__ == "__main__":
    main()