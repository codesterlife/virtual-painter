import cv2
import numpy as np
import HandTrackingModule as htm

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(5, 30)

detector = htm.handDetector()

drawing_color = (0, 0, 0)
imgCanvas = np.zeros((480, 640, 3), np.uint8)

erazerSize = 50
brushSize = 15

while True:
    #1. import image
    success, image = cap.read()

    image = cv2.flip(image, 1)

    cv2.rectangle(image, (0, 0), (1280, 60), (0, 0, 0), cv2.FILLED) # black background box for the colors

    cv2.rectangle(image, (10, 10), (80, 50), (0, 0, 255), cv2.FILLED) # red
    cv2.rectangle(image, (90, 10), (160, 50), (0, 255, 0), cv2.FILLED) # green
    cv2.rectangle(image, (170, 10), (240, 50), (255, 0, 0), cv2.FILLED) # blue
    cv2.rectangle(image, (250, 10), (320, 50), (0, 255, 255), cv2.FILLED) # yellow
    cv2.rectangle(image, (330, 10), (400, 50), (255, 0, 255), cv2.FILLED) # magenta
    cv2.rectangle(image, (410, 10), (480, 50), (255, 255, 0), cv2.FILLED) # cyan
    cv2.rectangle(image, (490, 10), (630, 50), (255, 255, 255), cv2.FILLED) # eraser box
    cv2.putText(image, 'Eraser', (505, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    

    #2. find landmarks

    image = detector.findHands(image)
    landmark_list = detector.findPosition(image)

    # print(landmark_list)

    if len(landmark_list) != 0:
        x1, y1 = landmark_list[8][1:] # coordinates of index finger
        x2, y2 = landmark_list[12][1:] # coordiates of middle finger

        # print(f"x1 {x1} y1 {y1} x2 {x2} y2 {y2}")

    #3. Check which finger is up

    fingers = detector.fingersUp()
    # print(fingers)

    #4. selection mode - two finger up condition

    if fingers[1] and fingers[2]:
        
        cv2.putText(image, "Selection Mode", (5, 475), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

        xp, yp = 0, 0
        
        if y1 < 70:
            
            if 10 < x1 < 80:
                drawing_color = (0, 0, 255)
                print("Red")

            if 90 < x1 < 160:
                drawing_color = (0, 255, 0)
                print("Green")

            if 170 < x1 < 240:
                drawing_color = (255, 0, 0)
                print("Blue")

            if 250 < x1 < 320:
                drawing_color = (0, 255, 255)
                print("Yellow")

            if 330 < x1 < 400:
                drawing_color = (255, 0, 255)
                print("Magenta")

            if 410 < x1 < 480:
                drawing_color = (255, 255, 0)
                print("Cyan")

            if 490 < x1 < 560:
                drawing_color = (0, 0, 0)
                print("Eraser")

        cv2.rectangle(image, (x1, y1), (x2, y2), drawing_color, cv2.FILLED)

    #5. drawing mode - one finger up condition

    if fingers[1] and not fingers[2]:

        cv2.putText(image, "Drawing", (5, 475), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

        cv2.circle(image, (x1, y1), 5, drawing_color, thickness = -1)

        if xp == 0 and yp == 0:
            
            xp = x1
            yp = y1

        if drawing_color == (0, 0, 0):

            cv2.line(image, (xp, yp), (x1, y1), drawing_color, erazerSize)
            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawing_color, erazerSize)

        else:

            cv2.line(image, (xp, yp), (x1, y1), drawing_color, brushSize)
            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawing_color, brushSize)

            xp, yp = x1, y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 20, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)

    image = cv2.bitwise_and(image, imgInv)
    image = cv2.bitwise_or(image, imgCanvas)

    image = cv2.addWeighted(image, 1, imgCanvas, 0.5, 0)

    cv2.imshow('virtual_painter', image)
    if cv2.waitKey(1) & 0xFF == 27:
        break