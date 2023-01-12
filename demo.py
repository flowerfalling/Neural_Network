# -*- coding: utf-8 -*-
# @Time    : 2023/1/10 19:57
# @Author  : 之落花--falling_flowers
# @File    : demo.py
# @Software: PyCharm
import numpy as np
from scipy import signal
import torch


def main():
    data = np.eye(7)
    conv = np.eye(3)
    r = signal.convolve2d(data, conv, 'valid')
    print(signal.convolve2d(r, conv[:, ::-1]))
    print(signal.convolve2d(data, r, 'valid'))
    a = torch.nn.MaxPool2d()
    pass


if __name__ == '__main__':
    main()
