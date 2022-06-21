import numpy as np
from pyrobot.brain import Brain

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from pyrobot.tools.followLineTools import findLineDeviation
from common import get_red_filtered_image, detect_arrow


class BrainFollowLine(Brain):
    NO_FORWARD = 0
    SLOW_FORWARD = 0.11
    MED_FORWARD = 0.5
    FULL_FORWARD = 0.7

    NO_TURN = 0
    MED_LEFT = 0.85
    HARD_LEFT = 1.8
    MED_RIGHT = -0.85
    HARD_RIGHT = -1.8

    NO_ERROR = 0

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

        # display the robot's camera's image using opencv
        cv2.imshow("Stage Camera Image", cv_image)
        cv2.waitKey(1)

        # write the image to a file, for debugging etc.
        # cv2.imwrite("debug-capture.png", cv_image)
        red_image = get_red_filtered_image(cv_image)

        # cv2.imshow("Red Stage Camera Image", red_img)
        # cv2.waitKey(1)

        arrow = detect_arrow(red_image)

        # convert the image into grayscale
        imageGray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # determine the robot's deviation from the line.
        foundLine, error = findLineDeviation(imageGray)
        # print("findLineDeviation returned ", foundLine, error)

        #     # display a debug image using opencv
        #     middleRowIndex = cv_image.shape[1]//2
        #     centerColumnIndex = cv_image.shape[0]//2
        #     if (foundLine):
        #       cv2.rectangle(cv_image,
        #                     (int(error*middleRowIndex)+middleRowIndex-5,
        #                      centerColumnIndex-5),
        #                     (int(error*middleRowIndex)+middleRowIndex+5,
        #                      centerColumnIndex+5),
        #                     (0,255,0),
        #                     3)
        #     cv2.imshow("Debug findLineDeviation", cv_image)
        #     cv2.waitKey(1)

        # A trivial on-off controller
        if (foundLine):
            # if(front and left and right > 0.5):
            if (error > 0.15 and error < 0.25):
                # print("Turning right.")
                self.move(self.MED_FORWARD, self.MED_RIGHT)
            elif (error < -0.15 and error > -0.25):
                # print("Turning left.")
                self.move(self.MED_FORWARD, self.MED_LEFT)
            elif (error > 0.25):
                self.move(self.SLOW_FORWARD, self.HARD_RIGHT)
            elif (error < -0.25):
                self.move(self.SLOW_FORWARD, self.HARD_LEFT)
            else:
                # print("Straight ahead.")
                self.move(self.FULL_FORWARD, self.NO_TURN)


def INIT(engine):
    assert (engine.robot.requires("range-sensor") and
            engine.robot.requires("continuous-movement"))

    return BrainFollowLine('BrainFollowLine', engine)