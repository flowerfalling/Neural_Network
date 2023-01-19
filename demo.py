# -*- coding: utf-8 -*-
# @Time    : 2023/1/10 19:57
# @Author  : 之落花--falling_flowers
# @File    : demo.py
# @Software: PyCharm
import numpy as np
import time
import matplotlib.pyplot as plt

import datasets


def main():
    s = datasets.MNIST()
    print(np.var(s.data / 255) - 0.13)
    ...


if __name__ == '__main__':
    main()
