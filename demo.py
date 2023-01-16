# -*- coding: utf-8 -*-
# @Time    : 2023/1/10 19:57
# @Author  : 之落花--falling_flowers
# @File    : demo.py
# @Software: PyCharm
import numpy as np
import time


def main():
    a = np.random.randn(10, 784)
    b = np.random.randn(200, 784)
    r = np.einsum('bi,oi->bo', a, b, optimize=True)
    ...


if __name__ == '__main__':
    main()
