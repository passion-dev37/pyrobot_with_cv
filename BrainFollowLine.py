import math

import numpy as np
from pyrobot.brain import Brain

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from pyrobot.tools.followLineTools import findLineDeviation
from common import get_red_filtered_image
from arrow_detection import filter_arrow
from junction_detection import detect_junction
from mark_detection import mark_recognization, init_mark_recognization_engine


class BrainFollowLine(Brain):
    NO_FORWARD = 0
    SLOW_FORWARD = 0.11
    MED_FORWARD = 0.43  # 0.5
    FULL_FORWARD = 0.63  # 0.7

    NO_TURN = 0
    MED_LEFT = 0.85
    HARD_LEFT = 1.8
    MED_RIGHT = -0.85
    HARD_RIGHT = -1.8

    MED_TURN_LEFT = 0.9
    HARD_TURN_LEFT = 2.05
    MED_TURN_RIGHT = -0.9
    HARD_TURN_RIGHT = -2.05

    TURN_ARG = 1.29

    NO_ERROR = 0
    check_turn = 0
    clfSeg, clfORB, clfHU = init_mark_recognization_engine()

    def setup(self):
        self.image_sub = rospy.Subscriber("/image", Image, self.callback)
        self.bridge = CvBridge()

    def callback(self, data):
        self.rosImage = data

    def destroy(self):
        cv2.destroyAllWindows()

    def step(self):
        # take the last image received from the camera and convert it into
        # opencv format
        try:
            cv_image = self.bridge.imgmsg_to_cv2(self.rosImage, "bgr8")
        except CvBridgeError as e:
            print(e)

        red_image = get_red_filtered_image(cv_image)
        # cv2.imshow("Red Image", red_image)
        # cv2.waitKey(1)

        arrow_image, exist, angle = filter_arrow(red_image)
        show_image = cv_image.copy()
        if not exist:
            show_image = mark_recognization(self.clfSeg, self.clfORB, self.clfHU, show_image)

        if exist:
            junc_exist, junc_center, junc_type = detect_junction(cv_image)
            if junc_exist:
                if junc_type:
                    cv2.putText(show_image, junc_type + " Junction Detected",
                            (50, 150), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 1)

        if exist:
            if angle < -90:
                angle += 360
            angle = angle - 90
            # print(angle)
        if exist and self.check_turn < 100 and abs(angle) > 10:
            self.check_turn = self.check_turn + 1

            # if (angle > 90):
            #     self.move(self.SLOW_FORWARD, self.HARD_TURN_LEFT)
            # elif (angle > 0):
            #     self.move(self.MED_FORWARD, self.MED_TURN_LEFT)
            # elif (angle < -90):
            #     self.move(self.SLOW_FORWARD, self.HARD_TURN_RIGHT)
            # else:
            #     self.move(self.MED_FORWARD, self.MED_TURN_RIGHT)

            if angle > 25:
                cv2.putText(show_image, "Left Arrow Detected",
                            (50, 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 1)
            elif angle < -25:
                cv2.putText(show_image, "Right Arrow Detected",
                            (50, 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 1)
        if exist and junc_exist:
            if abs(angle) > 90:
                forward = self.SLOW_FORWARD
            else:
                forward = self.MED_FORWARD

            turn = angle*math.pi*self.TURN_ARG/180
            self.move(forward, turn)
            self.move(forward, turn)
            # self.move(forward, turn)
        else:
            self.check_turn = 0
            # convert the image into grayscale
            imageGray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # determine the robot's deviation from the line.
            foundLine, error = findLineDeviation(imageGray)

            # A trivial on-off controller
            if (foundLine):
                # if(front and left and right > 0.5):
                if (error > 0.15 and error < 0.23):
                    # print("Turning right.")
                    self.move(self.MED_FORWARD, self.MED_RIGHT)
                elif (error < -0.15 and error > -0.23):
                    # print("Turning left.")
                    self.move(self.MED_FORWARD, self.MED_LEFT)
                elif (error > 0.23):
                    self.move(self.SLOW_FORWARD, self.HARD_RIGHT)
                elif (error < -0.23):
                    self.move(self.SLOW_FORWARD, self.HARD_LEFT)
                else:
                    # print("Straight ahead.")
                    self.move(self.FULL_FORWARD, self.NO_TURN)
        # display the robot's camera's image using opencv
        cv2.imshow("Stage Camera Image", show_image)
        cv2.waitKey(1)


def INIT(engine):
    assert (engine.robot.requires("range-sensor") and
            engine.robot.requires("continuous-movement"))

    return BrainFollowLine('BrainFollowLine', engine)