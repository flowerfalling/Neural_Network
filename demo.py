# -*- coding: utf-8 -*-
# @Time    : 2023/1/10 19:57
# @Author  : 之落花--falling_flowers
# @File    : demo.py
# @Software: PyCharm
import numpy as np
import scipy
from scipy import signal

import nn


class Conv2d(nn.Layer):
    def __init__(
            self,
            lr: float,
            in_channels: int,
            out_channels: int,
            kernel_size: int or tuple[int, int],
            stride: int or tuple[int, int] = (1, 1),
            padding: int = 0,
    ) -> None:
        self.lr = lr
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        self.w = np.random.randn(out_channels, in_channels, *self.kernel_size) / \
                 (self.kernel_size[0] * self.kernel_size[1])
        self.history_input = None
        self.loss = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.history_input = x
        x = np.pad(x, ((0, 0), (0, 0), *((self.padding,) * 2,) * 2))
        s = x.shape
        r = np.empty((s[0], self.out_channels,
                      (s[-2] - self.kernel_size[0] + 2 * self.padding) // self.stride[0] + 1,
                      (s[-1] - self.kernel_size[1] + 2 * self.padding) // self.stride[1] + 1))
        for b in range(s[0]):
            r[b] = signal.fftconvolve(x[b, None], self.w, 'valid')[:, 0]
        return r

    def backward(self, e: np.ndarray) -> np.ndarray:
        self.loss = e
        r = np.zeros(self.history_input.shape)
        for b in range(e.shape[0]):
            for c in range(e.shape[1]):
                r[b] += signal.fftconvolve(e[b, c, None], self.w[c, :, ::-1])
        return r if not self.padding else r[:, :, self.padding: -self.padding, self.padding: -self.padding]

    def step(self) -> None:
        for b in range(self.loss.shape[0]):
            self.w += signal.fftconvolve(self.loss[b, :, None], self.history_input[b, None], 'valid') * self.lr

    def __call__(self, x) -> np.ndarray:
        return self.forward(x)


class CNet(nn.Layer):
    def __init__(self, lr):
        self.lr = lr
        self.loss = []
        self.conv1 = Conv2d(self.lr, 1, 10, (3, 3))
        self.conv2 = Conv2d(self.lr, 10, 16, (4, 4))
        self.pool1 = nn.MaxPool2d((2, 2), (2, 2))
        self.pool2 = nn.MaxPool2d((2, 2), (2, 2))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.lr, 400, 120)
        self.fc2 = nn.Linear(self.lr, 120, 84)
        self.fc2 = nn.Linear(self.lr, 84, 10)
        self.sigmod = nn.Sigmod()
        self.relu1 = nn.Relu()
        self.relu2 = nn.Relu()
        self.relu3 = nn.Relu()
        self.relu4 = nn.Relu()

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.sigmod(self.fc3(x))
        return x

    def backward(self, e):
        pass

    def train(self, x, t):
        o = self(x)
        e = t - o
        self.loss.append((e ** 2).sum())
        e = self.sigmod4.backward(e)
        e = self.fc2.backward(e)
        e = self.sigmod3.backward(e)
        e = self.fc1.backward(e)
        e = self.flatten.backward(e)
        e = self.sigmod2.backward(e)
        e = self.pool2.backward(e)
        e = self.conv2.backward(e)
        e = self.sigmod1.backward(e)
        e = self.pool1.backward(e)
        self.conv1.backward(e)
        self.conv1.step()
        self.conv2.step()
        self.fc1.step()
        self.fc2.step()

    def __call__(self, x):
        return self.forward(x)


def main():
    data = np.random.randn(1, 3, 28, 28)
    r = np.random.randn(6, 1, 26, 26)
    one = np.ones((3, 4, 5, 6, 7))
    print((one * (1 - one)).shape)


if __name__ == '__main__':
    main()
