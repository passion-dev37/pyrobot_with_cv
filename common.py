import cv2
import numpy as np


def get_red_filtered_image(cv_image):
    img_hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

    # lower mask (0-10)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([170, 50, 50])
    upper_red = np.array([180, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    # join my masks
    mask = mask0 + mask1

    # set my output img to zero everywhere except my mask
    red_img = cv_image.copy()
    red_img[np.where(mask == 0)] = 0
    return cv2.cvtColor(red_img, cv2.COLOR_HSV2BGR)
