import cv2
import HandTrackingModules as htm
import HandTrackingModule


def isOpen(points, limit, img):
    up_4 = points[0][1:]
    up_3 = points[1][1:]
    up_2 = points[2][1:]
    up_1 = points[3][1:]

    cv2.line(img, up_4, up_3, (255, 0, 0), 5)
    cv2.line(img, up_2, up_3, (0, 255, 0), 5)
    cv2.line(img, up_2, up_1, (0, 0, 255), 5)

    font = cv2.FONT_HERSHEY_SIMPLEX
    try:
        m1 = round((up_3[1] - up_4[1]) / (up_3[0] - up_4[0]), 3)
    except:
        m1 = 1000
    try:
        m2 = round((up_3[1] - up_2[1]) / (up_3[0] - up_2[0]), 3)
    except:
        m2 = 1000
    try:
        m3 = round((up_2[1] - up_1[1]) / (up_2[0] - up_1[0]), 3)
    except:
        m3 = 1000
    try:
        m4 = round((up_4[1] - up_2[1]) / (up_4[0] - up_2[0]), 3)
    except:
        m4 = 1000
    try:
        m5 = round((up_3[1] - up_1[1]) / (up_3[0] - up_1[0]), 3)
    except:
        m5 = 1000

    cv2.putText(img, str(m1), up_3, font, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, str(m2), up_2, font, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, str(m3), up_1, font, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, str(m4), [10,50], font, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, str(m5), [200,50], font, 1, (0, 0, 0), 2, cv2.LINE_AA)

    #
    # if abs(m1 - m2) <= abs(limit * m1) and abs(m2 - m3) <= abs(limit * m2):
    #     return True
    # else:
    #     return False
    avg = (m1 + m2 + m3) / 3
    if abs((m1 - avg)/ avg) <= 0.5 and abs((m2 - avg)/ avg) <= 0.5 and abs((m3 - avg)/ avg)  <= 0.5:
        return True
    return False

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.HandDetector(min_detection_con=0.75)
tipIds = [4, 8, 12, 16, 20]
while True:
    success, img = cap.read()

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    # print(lmList)
    if len(lmList) != 0:
        fingers = []
        for i in range(1, 2):
            output = isOpen(lmList[4 * i + 1: 4 * (i + 1) + 1], 0.2, img)
            cv2.putText(img, str(output), [200, 200], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            fingers.append(output)
        print(fingers)
    # if len(lmList) != 0:
    #     fingers = []
    #     for id in range(0, len(tipIds)):
    #         if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
    #             fingers.append(1)
    #         else:
    #             fingers.append(0)
    #     print(fingers)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
