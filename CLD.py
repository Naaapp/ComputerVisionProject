import numpy as np
from ETF import ETF
from math import sqrt, pi
import numpy as np
from operator import add
from math import tanh
import cv2

class CLD:

    originalImg = np.zeros([200, 200], dtype=np.dtype('Float32'))
    result = np.zeros([200, 200], dtype=np.dtype('Float32'))
    DoG = np.zeros([200, 200], dtype=np.dtype('Float32'))
    FDoG = np.zeros([200, 200], dtype=np.dtype('Float32'))
    etf = ETF()

    sigma_m = 3.0
    sigma_c = 1.0
    rho = 0.997
    tau = 0.8

    def __init__(self, image):
        dims = image.shape
        self.width = dims[0]
        self.height = dims[1]
        self.originalImg = np.copy(image)
        self.result = np.zeros(dims, dtype=np.dtype('Float32'))
        self.DoG = np.zeros(dims, dtype=np.dtype('Float32'))
        self.FDoG = np.zeros(dims, dtype=np.dtype('Float32'))
        self.etf.initialisationETF(image)

    def gauss(self, x, mean, sigma):
        return 1. / (sqrt(2. * pi) * sigma) * np.exp(-np.power((x - mean) / sigma, 2.) / 2)


    # GAU will be a list

    def makeGaussianVector(self, sigma, GAU):
        threshold = 0.001
        i = 1

        while self.gauss(i, 0, sigma) > threshold:
            i = i + 1

        GAU.clear()
        GAU.append(self.gauss(0.0, 0.0, sigma))
        for j in range(i+1):
            GAU.append(self.gauss(j, 0, sigma))

    # 2 matrices images and two doubles
    def gradientDoG(self, src, dst, rho, sigma_c):

        SIGMA_RATIO = 0.6
        sigma_s = SIGMA_RATIO * sigma_c
        (gau_c, gau_s) = (list(), list())
        self.makeGaussianVector(sigma_c, gau_c)
        self.makeGaussianVector(sigma_s, gau_s)

        kernel = len(gau_s) - 1

        for x in range(dst.shape()[0]):
            for y in range(dst.shape()[1]):
                gau_c_acc = 0
                gau_s_acc = 0
                gau_c_weight_acc = 0
                gau_s_weight_acc = 0
                tmp = self.etf.flowField[x][y]
                gradient = tmp # mayby remove this tmp

                if gradient[0] == 0 and gradient[1] == 0:
                    continue

                for step in range(-kernel, kernel + 1):
                    row = x + gradient[0] * step
                    col = y + gradient[1] * step

                    if col > dst.shape()[1] - 1 or col < 0.0 or row > dst.shape()[0] - 1 or row < 0.0:
                        continue

                    value = src[int(row)][int(col)]

                    gau_idx = abs(step)

                    if gau_idx >= len(gau_c):
                        gau_c_weight = 0.0
                    else:
                        gau_c_weight = gau_c[gau_idx]

                    gau_s_weight = gau_s[gau_idx]

                    gau_c_acc = gau_c_acc + value * gau_c_weight
                    gau_s_acc = gau_s_acc + value * gau_s_weight
                    gau_c_weight_acc = gau_c_weight_acc + gau_c_weight
                    gau_s_weight_acc = gau_s_weight_acc + gau_s_acc

                v_c = gau_c_acc / gau_c_weight_acc
                v_s = gau_s_acc / gau_s_weight_acc
                dst[x][y] = v_c - rho * v_s

    def flowDoG(self, src, dst, sigma_m):

        gau_m = list()
        self.makeGaussianVector(sigma_m, gau_m)

        img_h = src.shape()[0]
        img_w = src.shape()[1]
        kernel_half = len(gau_m) - 1

        for x in range(0, img_h):
            for y in range (0, img_w):
                gau_m_acc = -gau_m[0] * src[x][y]
                gau_m_weight_acc = -gau_m[0]

                pos = [x, y]

                for step in range (0, kernel_half):
                    tmp = self.etf.flowField[int(x)][int(y)]
                    direction = [tmp[1], tmp[0]]

                    if direction[0] == 0 and direction[1] == 0:
                        break
                    if pos[0] > img_h - 1 or pos[0] < 0 or pos[1] > img_w - 1 or pos[1]:
                        break

                    value = src[int(pos[0])][int(pos[1])]
                    weight = gau_m[step]

                    gau_m_acc = gau_m_acc + value * weight
                    gau_m_weight_acc = gau_m_weight_acc + weight
                    # Add both list
                    pos = list(map(add, pos, direction))

                    if int(pos[0]) < 0 or int(pos[0]) > img_h - 1 or int(pos[1]) < 0 or int(pos[1]) > img_w - 1:
                        break

                if gau_m_acc / gau_m_weight_acc > 0:
                    dst[x][y] = 1.0
                else:
                    dst[x][y] = 1 + tanh(gau_m_acc / gau_m_weight_acc)

        cv2.normalize(dst, dst, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)

    def genCLD(self):

        self.gradientDoG(self.originalImg, self.DoG, self.rho, self.sigma_c)
        self.flowDoG(self.DoG, self.flowDoG(), self.sigma_m)
        self.binaryThresholding(self.FDoG, self.result, self.tau)

    def genCLD_Iter(self, refiningSteps, ETF_Kernel, iterativeSteps):

        if iterativeSteps <= 0:
            return

        for i in range(refiningSteps):
            self.etf.refineETF(ETF_Kernel)

        self.genCLD()

        for i in range(0, iterativeSteps - 1):
            self.combineImage()
            self.genCLD()

    def binaryThresholding(self, src, dst, tau):
        for x in range(self.width):
            for y in range(self.height):
                if src[x][y] < tau:
                    dst[x][y] = 0.0
                else:
                    dst[x][y] = 255.0

    def combineImage(self):

        for x in range(self.width):
            for y in range(self.height):
                if self.result[x][y] == 0:
                    self.originalImg[x][y] = 0
