import math
from math import sqrt, atan

import cv2
import HandTrackingModules as htm
import HandTrackingModule


def findTriArea(org, point):
    if org[0] <= point[0] and org[1] < point[1]:
        return 1
    if org[0] > point[0] and org[1] <= point[1]:
        return 2
    if org[0] >= point[0] and org[1] > point[1]:
        return 3
    if org[0] < point[0] and org[1] >= point[1]:
        return 4


def isInside(r, teta, p):
    x = p[0]
    y = p[1]
    length = sqrt(x ** 2 + y ** 2)
    if length > r:
        return False
    area = findTriArea([0, 0], p)
    alpha = (math.pi / 2) * area
    if x != 0:
        alpha = atan(y / x)
        if alpha == 0:
            if area == 2:
                alpha = math.pi
            elif area == 4:
                alpha = math.pi * 2
        else:
            if area == 2 or area == 3:
                alpha += math.pi
            elif area == 4:
                alpha += 2 * math.pi
    low_limit = (teta - math.pi / 6)
    high_limit = (teta + math.pi / 6)
    if (1.5 * math.pi < alpha <= math.pi * 2) and low_limit <= 0:
        low_limit += 2 * math.pi
        high_limit += 2 * math.pi
    if (1.5 * math.pi < teta <= math.pi * 2) and area == 1:
        alpha += 2 * math.pi
    if low_limit < alpha < high_limit:
        return True
    return False


def isOpen(points, limit, img, org):
    height, width = img.shape[0], img.shape[1]
    up_1 = points[0][1:]
    up_2 = points[1][1:]
    up_3 = points[2][1:]
    up_4 = points[3][1:]

    up_1[1] = height - up_1[1]
    up_2[1] = height - up_2[1]
    up_3[1] = height - up_3[1]
    up_4[1] = height - up_4[1]
    org[1] = height - org[1]

    font = cv2.FONT_HERSHEY_SIMPLEX
    area = findTriArea(up_1, up_2)
    if up_1[0] != up_2[0]:
        m1 = (up_2[1] - up_1[1]) / (up_2[0] - up_1[0])
        if m1 == 0:
            if area == 2:
                teta = math.pi
            if area == 4:
                teta = math.pi * 2
        else:
            teta = atan(m1)
            if area == 2 or area == 3:
                teta += math.pi
            elif area == 4:
                teta += 2 * math.pi
    else:
        teta = area * (math.pi / 2)

    l = round(sqrt((up_2[1] - up_1[1]) ** 2 + (up_2[0] - up_1[0]) ** 2), 3)
    r = 2.5 * l

    x1 = up_1[0] - up_2[0]
    x2 = up_2[0] - up_3[0]
    x3 = up_3[0] - up_4[0]
    x = (x1 ** 2) + (x2 ** 2) + (x3 ** 2)
    y1 = up_1[1] - up_2[1]
    y2 = up_2[1] - up_3[1]
    y3 = up_3[1] - up_4[1]
    y = (y1 ** 2) + (y2 ** 2) + (y3 ** 2)
    sgn = False

    if x > y:
        if x1 * x2 > 0 and x2 * x3 > 0:
            if (x1 * (org[0] - up_1[0])) < 0:
                return False
            sgn = True
        else:
            return False
    if y > x:
        if y1 * y2 > 0 and y2 * y3 > 0:
            if (y1 * (org[1] - up_1[1])) < 0:
                return False
            sgn = True
        else:
            return False

    a = isInside(r, teta, [up_3[0] - up_2[0], up_3[1] - up_2[1]])
    b = isInside(r, teta, [up_4[0] - up_2[0], up_4[1] - up_2[1]])

    # cv2.putText(img, str(l) + " " + str(round((teta / math.pi) * 180)), (100, 100), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
    # cv2.putText(img, str(a) + " " + str(b), (10, 150), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
    return a and b and sgn


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
        for i in range(0, 5):
            output = isOpen(lmList[4 * i + 1: 4 * (i + 1) + 1], 0.2, img, lmList[0][1:])
            # cv2.putText(img, str(output), [200, 200], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            fingers.append(output)
        a = 0
        for i in fingers:
            if i:
                a += 1
        cv2.putText(img, str(a), [10, 100], cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5, cv2.LINE_AA)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
