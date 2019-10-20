import numpy as np
from ETF import ETF
from math import sqrt, pi

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

    def __init__(self):
        pass

    def gauss(self, x, mean, sigma):
        return 1. / (sqrt(2. * pi) * sigma) * np.exp(-np.power((x - mean) / sigma, 2.) / 2)