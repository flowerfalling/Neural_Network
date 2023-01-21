# -*- coding: utf-8 -*-
# @Time    : 2023/1/10 19:57
# @Author  : 之落花--falling_flowers
# @File    : demo.py
# @Software: PyCharm
# import torch
import numpy as np

import datasets
import nn
import nndemo_mnist


def main():
    net = nndemo_mnist.Net()
    print(net.parameters())
    ...


if __name__ == '__main__':
    main()
