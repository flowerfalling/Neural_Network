# -*- coding: utf-8 -*-
# @Time    : 2023/1/10 19:57
# @Author  : 之落花--falling_flowers
# @File    : demo.py
# @Software: PyCharm
# import torch
import numpy as np
from scipy import signal
from numpy.lib import stride_tricks


# import nn


def main():
    x = np.random.randint(0, 10, (1, 1, 11, 11))
    r = stride_tricks.sliding_window_view(x, (2, 2), axis=(2, 3))[:, :, ::2, ::3]
    print(x)
    print(r)


if __name__ == '__main__':
    main()
