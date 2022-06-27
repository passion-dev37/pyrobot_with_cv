import math

import cv2
import numpy
import numpy as np
# Image path


def get_blue_filtered_image(cv_image):
    # Define lower and upper limits of our blue
    BlueMin = np.array([60, 35, 140], np.uint8)
    BlueMax = np.array([120, 255, 255], np.uint8)

    # Go to HSV colourspace and get mask of blue pixels
    HSV = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(HSV, BlueMin, BlueMax)

    # Make all pixels in mask white
    result = np.zeros_like(cv_image)
    result[mask > 0] = [255, 255, 255]
    return result


def get_angle(points0, points1):
    dx, dy = points1[0] - points0[0], points1[1] - points0[1]
    rads = math.atan2(-dy, dx)
    degrees = math.degrees(rads)
    if degrees < 0:
        degrees += 360
    return degrees


def get_length(p1, p2):
    line_length = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
    return line_length

    
def get_directions(x, y, x_list, y_list):
    result_points = []
    result_degrees = []
    for i in range(len(x_list)):
        distance = get_length((x, y), (x_list[i], y_list[i]))
        # print(distance)
        if distance < 50:
            continue
        angle = get_angle((x, y), (x_list[i], y_list[i]))
        if len(result_degrees) == 0:
            result_points.append((x_list[i], y_list[i]))
            result_degrees.append(angle)
        else:
            add = True
            for j in range(len(result_degrees)):
                diff = abs(result_degrees[j] - angle)
                if diff > 180:
                    diff = 360 - diff
                if diff <= 45:
                    distance1 = get_length((x, y), (x_list[i], y_list[i]))
                    distance2 = get_length((x, y), result_points[j])
                    if distance1 > distance2:
                        result_points[j] = (x_list[i], y_list[i])
                        result_degrees[j] = angle
                    add = False
                    break
            if add:
                result_points.append((x_list[i], y_list[i]))
                result_degrees.append(angle)
    return result_points, result_degrees


def detect_junction(inputImage):
    # Prepare a deep copy of the input for results:
    inputImageCopy = get_blue_filtered_image(inputImage)
    kernel = np.ones((25, 25), np.float32)/25
    processImage = cv2.filter2D(inputImageCopy, -1, kernel)
    # cv2.imshow("Processed", processImage)
    # cv2.waitKey(1)

    # Grayscale conversion:
    grayscaleImage = cv2.cvtColor(processImage, cv2.COLOR_BGR2GRAY)

    # Add borders to prevent skeleton artifacts:
    borderThickness = 1
    borderColor = (0, 0, 0)
    grayscaleImage = cv2.copyMakeBorder(grayscaleImage, borderThickness, borderThickness, borderThickness,
                                        borderThickness,
                                        cv2.BORDER_CONSTANT, None, borderColor)
    # Compute the skeleton:
    skeleton = cv2.ximgproc.thinning(grayscaleImage, None, 1)

    cv2.imshow("Skeleton", skeleton)
    # cv2.waitKey(1)

    # Threshold the image so that white pixels get a value of 10 and
    # black pixels a value of 0:
    _, binaryImage = cv2.threshold(skeleton, 128, 10, cv2.THRESH_BINARY)

    # Set the intersections kernel:
    h = np.array([[1, 1, 1],
                  [1, 10, 1],
                  [1, 1, 1]])

    # Convolve the image with the kernel:
    imgFiltered = cv2.filter2D(binaryImage, -1, h)

    # Prepare the final mask of points:
    (height, width) = binaryImage.shape
    pointsMask = np.zeros((height, width, 1), np.uint8)

    # Perform convolution and create points mask:
    thresh = 130
    # Locate the threshold in the filtered image:
    pointsMask = np.where(imgFiltered == thresh, 255, 0)

    # Convert and shape the image to a uint8 height x width x channels
    # numpy array:
    pointsMask = pointsMask.astype(np.uint8)
    pointsMask = pointsMask.reshape(height, width, 1)

    # Set kernel (structuring element) size:
    kernelSize = 7
    # Set operation iterations:
    opIterations = 3
    # Get the structuring element:
    morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))

    # Perform Dilate:
    pointsMask = cv2.morphologyEx(pointsMask, cv2.MORPH_DILATE, morphKernel, None, None, opIterations,
                                  cv2.BORDER_REFLECT101)

    # Get the coordinates of the end-points:
    (Y, X) = np.where(pointsMask == 255)
    if len(X) > 0 and len(Y) > 0:
        # Get the centroid:
        y = int(np.mean(Y))
        x = int(np.mean(X))

        if y < height/8 or y > height*7/8:
            return False, None
        if x < width/8 or x > width*7/8:
            return False, None

        # directions, angles = get_directions(x, y, X, Y)
        # print("Directions")
        # print(len(directions))
        # print(angles)
        # Draw the intersection point:
        # Set circle color:
        color = (0, 0, 255)

        # Draw Circle
        cv2.circle(inputImageCopy, (x, y), 3, color, -1)

        # Show Image
        cv2.imshow("Intersections", inputImageCopy)
        cv2.waitKey(1)
        return True, (x, y)

    return False, None
