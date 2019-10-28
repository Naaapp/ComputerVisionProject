# Edge Tangent Flow module for the Coherent Line Drawing

import cv2
import numpy as np
import math
from sklearn.preprocessing import normalize


class ETF:

    size = list()

    src = np.zeros([200, 200])
    src_n = np.zeros([200, 200])

    # matices of same size
    flowField = np.empty([200, 200]) # Matrix of 2d vector (array)
    refinedETF = np.zeros([200, 200])
    gradientMag = np.zeros([200, 200])

    def __init__(self):
        pass

    # initiation of the ETF field from an image by using the Sobel operator (image as narray as input)
    def initialisationETF(self, image):
        # Assumption of a 2D image
        self.size.append(image.shape[0])
        self.size.append(image.shape[1])
        self.src = np.copy(image)
        self.src_n = self.normalizeInput(np.copy(image))

        # Resizing of the matrices
        self.flowField = np.zeros(self.size + [2], dtype=np.dtype('Float32'))
        self.refinedETF = np.zeros(self.size + [2], dtype=np.dtype('Float32'))
        self.gradientMag = np.zeros(self.size, dtype=np.dtype('Float32'))

        # Use of Sobel to determined magnetude
        gradX = cv2.Sobel(self.src, cv2.CV_32F, 1, 0, ksize=5)
        gradY = cv2.Sobel(self.src, cv2.CV_32F, 0, 1, ksize=5)

        # Compute gradient
        cv2.magnitude(gradX, gradY, self.gradientMag)
        self.gradientMag = self.normalizeInput(self.gradientMag)

        for i in range(self.size[0]):
            for j in range(self.size[1]):
                cv2.normalize(np.array([gradX[i, j], gradY[i, j]]), self.flowField[i, j], alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)

        self.rotate(self.flowField, 90)


    def rotate(self, input, degree):
        rad = np.radians(degree)
        f = lambda x: np.array(
            [x[0] * math.cos(rad) - x[1] * math.sin(rad), x[1] * math.cos(rad) + x[0] * math.sin(rad)])
        for x in range(self.size[0]):
            for y in range(self.size[1]):
                input[x][y] = f(input[x][y])


    def normalizeInput(self, input):

        #return normalize(input[:, np.newaxis], axis=0).ravel()
        return normalize(input, axis=1, norm='max')

    def refineETF(self, kernel):
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                tmp = self.computeNewVector(i, j, kernel)
                self.refinedETF[i][j][0] = tmp[0]
                self.refinedETF[i][j][1] = tmp[1]

        self.flowField = np.copy(self.refinedETF)

    # paper eq 1 (int,int,int)
    def computeNewVector(self, x, y, kernel):

        tCur = self.flowField[x][y]

        for r in range(y - kernel, y + kernel + 1):
            if r < 0 or r >= self.size[1]:
                continue
            for c in range(x - kernel, x + kernel + 1):
                if c < 0 or c >= self.size[0]:
                    continue

                tCurNeigh = self.flowField[c][r]

                phi = self.computePhi(tCur, tCurNeigh)
                ws = self.computeWs(np.array([x, y]), np.array([c, r]), kernel)
                wm = self.computeWm(np.linalg.norm(self.gradientMag[x][y]), np.linalg.norm(self.gradientMag[c][r]))
                wd = self.computeWd(tCur, tCurNeigh)

                actualNeighb = phi * ws * wm * wd * tCurNeigh
                tCur = np.add(tCur, actualNeighb)

        if (abs(tCur[0]) + abs(tCur[1])) != 0:
            tCur[0] = tCur[0] / (abs(tCur[0]) + abs(tCur[1]))
            tCur[1] = tCur[1] / (abs(tCur[0]) + abs(tCur[1]))
        return tCur



    # paper eq 5 (Vec3f, Vec3f)
    def computePhi(self, xvect, yvect):
        if np.dot(xvect,yvect) > 0:
            return 1
        else:
            return -1

    # paper eq 2 (point2F, point2F, int)
    def computeWs(self, a, b, radius):
        euclideanDist = math.sqrt(np.dot(a,b))

        if euclideanDist < radius:
            return 1
        else:
            return 0

    # paper eq 3 (float, float)
    def computeWm(self, gradmag_x, gradmag_y):
        return (1 + np.tanh(gradmag_x - gradmag_y)) / 2

    # paper eq 4 (vct3f, vct3f)
    def computeWd(self, xvect, yvect):
        return abs(np.dot(xvect, yvect))

