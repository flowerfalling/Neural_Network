# -*- coding: utf-8 -*-
# @Time    : 2023/1/10 19:57
# @Author  : 之落花--falling_flowers
# @File    : demo.py
# @Software: PyCharm
import numpy as np
import cnn_mnist
import torch
from scipy.signal import convolve2d
from scipy.signal import convolve


def main():
    # data = np.zeros((3, 784))
    # fc1 = cnn_mnist.Linear(0.1, 784, 1)
    # print(fc1(data))
    # data = np.ones((1, 30, 30))
    # conv = np.ones((1, 3, 3))
    # c1 = cnn_mnist.Conv2d(0.1, 3, 6, (3, 3))
    #
    # print(convolve(data, conv, 'valid').shape)
    # data = np.array(np.eye(7) + np.eye(7)[:, ::-1])
    # nn = cnn_mnist.Conv2d(0.1, 3, 6, (3, 3))
    # nn.w = np.array([[np.eye(3) + np.eye(3)[:, ::-1] for _ in range(3)] for _ in range(6)])
    # print(nn.backward(nn(data)).shape)
    # print(data)
    # print(data[3, data[3] < 1])

    ts = np.eye(10)[:, None]
    print(ts[np.array([1, 2, 3, 3, 2, 1])].shape)


if __name__ == '__main__':
    main()
