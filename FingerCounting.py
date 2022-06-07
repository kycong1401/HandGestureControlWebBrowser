import cv2 as cv
import time
import os
import HandTrackingModule as htm
wCam, hCam = 640, 480

cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = "Finger"
myList = os.listdir(folderPath)
print(myList)

overlayList = []
for imPath in myList:
    image = cv.imread(f'{folderPath}/{imPath}')
    #print(f'{folderPath}/{imPath}')
    overlayList.append(image)

print(len(overlayList))

detector = htm.handDetector(detectionCon=0.75)

tipIDs = [4, 8, 12, 16, 20]

while True:
    success, image = cap.read()
    image = detector.findHands(image)
    lmList = detector.findPosition(image)
    #print(lmList)


    if len(lmList) != 0:
        fingers = []

        #Thumb
        if lmList[tipIDs[0]][1] > lmList[tipIDs[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        #Fingers
        for id in range(1, 5):
            if lmList[tipIDs[id]][2] < lmList[tipIDs[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        #print(fingers)
        totalFingers = fingers.count(1)
        print(totalFingers)

        cv.rectangle(image, (0, 250), (250, 500), (0, 255, 0), cv.FILLED)
        cv.putText(image, str(totalFingers), (45,460), cv.FONT_HERSHEY_PLAIN, 10,(255,0,0), 25)


        h, w, c = overlayList[totalFingers-1].shape
        image[0:h, 0:w] = overlayList[totalFingers-1]

    cv.imshow("Image", image)
    key = cv.waitKey(1)


#End