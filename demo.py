# -*- coding: utf-8 -*-
# @Time    : 2023/1/10 19:57
# @Author  : 之落花--falling_flowers
# @File    : demo.py
# @Software: PyCharm
import numpy as np
from scipy import signal


def main():
    data = np.eye(7)
    conv = np.eye(3)
    print(signal.convolve(data, conv, 'valid'))
    pass


if __name__ == '__main__':
    main()
