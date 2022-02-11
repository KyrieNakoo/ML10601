import csv
import time
import scipy.linalg as la
import numpy as np
from numpy import *


def SGD(ini, theta, label, x):
    a = 1
    # xi = word_matrix[:, entry_index]
    N = len(x.T)
    # print(theta.T@xi)
    # theta = a * xi/N*(label[entry_index] - math.exp(dot(theta.T, xi))/(1+math.exp(dot(theta.T, xi))))
    for j in range(len(x.T)):
        ini += x[:, j]/N*(-label[j] + math.exp(dot(theta.T, x[:, j]))/(1+math.exp(dot(theta.T, x[:, j]))))
        # print(a, a.shape)
    theta -= a*ini
    print(ini, '\n','\n', theta)


if __name__ == '__main__':
    x = mat([[0,0,1,0,1], [0,1,0,0,0], [0,1,1,0,0], [1,0,0,1,0]]).T
    label = mat([[0,1,1,0]]).T
    theta = mat([[1.5,2,1,2,3]]).T
    ini = mat(np.zeros([1,5])).T
    # print(x[:,1], label.shape, theta.shape, ini.shape)
    SGD(ini, theta, label, x)

