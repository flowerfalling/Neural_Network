# -*- coding: utf-8 -*-
# @Time    : 2023/1/10 19:57
# @Author  : 之落花--falling_flowers
# @File    : demo.py
# @Software: PyCharm
# import torch
import numpy as np
from scipy import signal
from numpy.lib import stride_tricks


import nn


def main():
    x = np.array([[1]])
    print(np.pad(x, ((1, 2), (3, 4))))
    pass


if __name__ == '__main__':
    main()
