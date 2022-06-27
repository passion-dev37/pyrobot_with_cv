import math

import cv2
import numpy as np

def preprocess(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
    img_canny = cv2.Canny(img_blur, 50, 50)
    kernel = np.ones((3, 3))
    img_dilate = cv2.dilate(img_canny, kernel, iterations=2)
    img_erode = cv2.erode(img_dilate, kernel, iterations=1)
    return img_erode

def get_length(p1, p2):
    line_length = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
    return line_length

def find_tip(points, convex_hull):
    length = len(points)
    max_distance = 0
    max_k = 0
    indices = np.setdiff1d(range(length), convex_hull)

    for i in range(2):
        j = indices[i] + 2
        if j > length - 1:
            j = length - j
        if np.all(points[j] == points[indices[i - 1] - 2]):
            for k in range(length):
                distance = get_length(points[j], points[k])

                if distance > max_distance:
                    max_distance = distance
                    max_k = k

            dx, dy = points[j][0] - points[max_k][0], points[j][1] - points[max_k][1]
            rads = math.atan2(-dy, dx)
            degrees = math.degrees(rads)
            return tuple(points[j]), degrees
    return None, None

def filter_arrow(img):
    contours, hierarchy = cv2.findContours(preprocess(img), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    result_image = np.zeros_like(img)
    count = 0
    angle = None
    height = len(img)
    # width = len(img[0])
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)
        hull = cv2.convexHull(approx, returnPoints=False)
        sides = len(hull)
        if 6 > sides > 3 and sides + 2 == len(approx):
            arrow_tip, degrees = find_tip(approx[:, 0, :], hull.squeeze())
            if arrow_tip:
                (x, y) = arrow_tip
                if y < height / 8 or y > height * 7 / 8:
                    return result_image, False, None
                count = count + 1
                angle = degrees
                cv2.drawContours(result_image, [cnt], -1, (0, 255, 0), 3)
                cv2.circle(result_image, arrow_tip, 3, (0, 0, 255), cv2.FILLED)
    return result_image, count > 0, angle
