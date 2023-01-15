# -*- coding: utf-8 -*-
# @Time    : 2023/1/10 19:57
# @Author  : 之落花--falling_flowers
# @File    : demo.py
# @Software: PyCharm
import numpy as np

import nn


def main():
    d = np.random.rand(784, 1)
    fc = np.random.randn(200, 784) / 784 * 50
    r = np.dot(fc, d)
    print(r.mean())
    print(r.var())
    ...


if __name__ == '__main__':
    main()
