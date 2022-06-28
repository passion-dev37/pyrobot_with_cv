import cv2
import os
import sys
from time import time
import numpy as np
import imageio as iio
from imageio import imread
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
# Image path

# Classifier for the segmentation
def init_mark_recognization_engine():
    # We create the classifier
    origImg = iio.v3.imread('imgs/original.png')
    markImg = iio.v3.imread('imgs/marked.png')
    clfSeg = NearestCentroid()
    data, lbls = None, None

    #  for i, j in zip(sorted(os.listdir(imgPath)), sorted(os.listdir(mkImgPath))):
    imNp = cv2.cvtColor(origImg, cv2.COLOR_BGR2RGB)
    markImg = cv2.cvtColor(markImg, cv2.COLOR_BGR2RGB)

    # We prepare the training data
    # we obtain all the points marked red, green, blue
    data_marca = imNp[np.all(markImg == [255, 0, 0], 2)]
    data_fondo = imNp[np.all(markImg == [0, 255, 0], 2)]
    data_linea = imNp[np.all(markImg == [0, 0, 255], 2)]

    # We prepare the labels
    lbl_marca = np.zeros(data_marca.shape[0], dtype=np.uint8)
    lbl_fondo = np.ones(data_fondo.shape[0], dtype=np.uint8)
    lbl_linea = np.zeros(data_linea.shape[0], dtype=np.uint8) + 2

    if data is None or lbls is None:
        data = np.concatenate([data_marca, data_fondo, data_linea])
        lbls = np.concatenate([lbl_marca, lbl_fondo, lbl_linea])
    else:
        data = np.concatenate([data, np.concatenate([data_marca, data_fondo, data_linea])])
        lbls = np.concatenate([lbls, np.concatenate([lbl_marca, lbl_fondo, lbl_linea])])

    clfSeg.fit(data, lbls)

    # ORB Descriptors
    path = os.getcwd() + '/marks'
    size = len(os.listdir(path))
    etiq = 0

    data, lbls = np.empty((size, 256)), np.empty(size)
    for i, img in enumerate(sorted(os.listdir(path))):
        imNp = cv2.cvtColor(cv2.imread(path + '/' + img), cv2.COLOR_BGR2RGB)
        predImg = clfSeg.predict(np.reshape(imNp, (imNp.shape[0] * imNp.shape[1], imNp.shape[2])))
        predImg = np.reshape(predImg, (imNp.shape[0], imNp.shape[1]))
        linImg = (predImg == 0).astype(np.uint8) * 255  # 0 es el valor del label marca

        contours, _ = cv2.findContours(linImg, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        orb = cv2.ORB_create()
        elipse = cv2.fitEllipse(max(contours, key=lambda x: len(x)))
        c, ej, ang = np.array(elipse[0]), np.array(elipse[1]), elipse[2]
        if ang > 90:
            ang -= 180
        lkp, des = orb.compute(linImg, [cv2.KeyPoint(c[0], c[1], np.mean(ej) * 1.3, ang - 90)])
        res = np.unpackbits(des).T

        data[i] = res
        lbls[i] = etiq
        etiq += (((i + 1) % 7) == 0)

        # # We paint the detection result
        # cv2.drawContours(imNp, contours, -1, (0, 255, 0), 2)
        # cv2.ellipse(imNp, elipse, (255, 0, 255), 2)
        # imDraw = cv2.drawKeypoints(imNp, lkp, None, color=(0, 0, 255), flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
        # # cv2.imwrite("marked_image.png", imDraw)
        # plt.imshow(imDraw)

    #ORB Descriptor classifier
    clfORB = KNeighborsClassifier(1, metric="euclidean")
    clfORB.fit(data,lbls)

    # HU Moments calculation
    etiq = 0
    path = os.getcwd() + '/marks'
    size = len(os.listdir(path))
    data, lbls = np.empty((size, 7)), np.empty(size)
    for i, img in enumerate(sorted(os.listdir(path))):
        imNp = cv2.cvtColor(cv2.imread(path + '/' + img), cv2.COLOR_BGR2RGB)
        predImg = clfSeg.predict(np.reshape(imNp, (imNp.shape[0] * imNp.shape[1], imNp.shape[2])))
        predImg = np.reshape(predImg, (imNp.shape[0], imNp.shape[1]))
        linImg = (predImg == 0).astype(np.uint8) * 255

        contours, _ = cv2.findContours(linImg, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        contours = max(contours, key=lambda x: len(x))

        data[i] = cv2.HuMoments(cv2.moments(contours, True)).T
        lbls[i] = etiq

        etiq += (((i + 1) % 7) == 0)

    # HU Moments classifier
    clfHU = KNeighborsClassifier(1, metric="euclidean")
    clfHU.fit(data, lbls)

    return clfSeg, clfORB, clfHU


def mark_recognization(clfSeg, clfORB, clfHU, frame):
    tags = {0: "Man", 1: "Stairs", 2: "Telephone", 3: "Woman"}
    imNp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    predicted_image = clfSeg.predict(np.reshape(imNp, (imNp.shape[0] * imNp.shape[1], imNp.shape[2])))
    predImg = np.reshape(predicted_image, (imNp.shape[0], imNp.shape[1]))  # Recuperamos las dimensiones
    paleta = np.array([(255, 0, 0), (0, 255, 0), (0, 0, 255)], dtype=np.uint8)

    # salidas, centro = bif.existen_bifurcaciones(frame, predImg, centro)
    # if len(salidas) <= 1:

    linImg = (predImg == 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(linImg, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    if len(contours) > 0:
        contours = max(contours, key=lambda x: len(x))

        if not len(contours) < 100:
            try:
                # Calculo descritprores ORB
                orb = cv2.ORB_create()
                elipse = cv2.fitEllipse(contours)
                c, ej, ang = np.array(elipse[0]), np.array(elipse[1]), elipse[2]
                if ang > 90:
                    ang -= 180
                _, des = orb.compute(linImg, [cv2.KeyPoint(c[0], c[1], np.mean(ej) * 1.3, ang - 90)])
                res = np.unpackbits(des).T

                fig = clfORB.predict([res])
                cv2.putText(frame, 'Identificado ORB {} '.format(tags[fig[0]]), (15, 25), cv2.FONT_HERSHEY_TRIPLEX,
                            0.55, (0, 0, 255))

                # Calculo momentos HU
                fig = clfHU.predict(cv2.HuMoments(cv2.moments(contours, True)).T)
                cv2.putText(frame, 'Identificado Hu {} '.format(tags[fig[0]]), (15, 50), cv2.FONT_HERSHEY_TRIPLEX,
                            0.55, (0, 0, 255))

            except:
                pass

    return frame
