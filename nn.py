# -*- coding: utf-8 -*-
# @Time    : 2023/1/13 10:47
# @Author  : 之落花--falling_flowers
# @File    : nn.py
# @Software: PyCharm
from abc import ABCMeta, abstractmethod

import numpy as np
import scipy
from scipy import signal


class Layer(metaclass=ABCMeta):
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, e):
        pass


class Linear(Layer):
    def __init__(self, lr, input_size, output_size):
        self.lr = lr
        self.h = None
        self.e = None
        self.w = np.random.rand(output_size, input_size) - 0.5
        self.b = np.random.rand(output_size) - 0.5

    def forward(self, x):
        self.h = x
        r = np.empty((x.shape[0], *self.b.shape))
        for b in np.arange(x.shape[0]):
            r[b] = np.dot(self.w, x[b]) + self.b
        return r

    def backward(self, e):
        self.e = e
        r = np.empty(self.h.shape)
        for b in np.arange(e.shape[0]):
            r[b] = np.dot(self.w.T, e[b])
        return r

    def step(self):
        for b in np.arange(self.e.shape[0]):
            self.w += np.dot(self.e[b, None].T, self.h[b, None]) * self.lr
            self.b += self.e[b] * self.lr

    def __call__(self, x):
        return self.forward(x)


class Conv2d(Layer):
    def __init__(self, lr, in_channels, out_channels, kernel_size, stride=(1, 1), padding=0):
        self.lr = lr
        self.h = None
        self.e = None
        self.stride = stride
        self.padding = padding
        self.w = np.random.rand(out_channels, in_channels, kernel_size[0], kernel_size[1]) - 0.5

    def forward(self, x):
        self.h = x
        r = [[0 for _ in range(self.w.shape[0])] for _ in range(x.shape[0])]
        for b in range(x.shape[0]):
            for t in range(self.w.shape[0]):
                r[b][t] = signal.convolve(x[b], self.w[t], 'valid')
        return np.array(r)[:, :, 0]

    def backward(self, e):
        self.e = e
        r = np.empty(self.h.shape)
        for b in np.arange(e.shape[0]):
            for t in np.arange(self.w.shape[1]):
                pad = (self.w.shape[-2] - self.stride[0], self.w.shape[-1] - self.stride[1])
                r[b, t] = signal.convolve(np.pad(e[b], ((0, 0), (pad[0], pad[0]), (pad[1], pad[1]))),
                                          self.w[:, t, ::-1], 'valid')
        return r

    def step(self):
        for b in np.arange(self.e.shape[0]):
            for o in np.arange(self.w.shape[0]):
                for i in np.arange(self.w.shape[1]):
                    self.w[o, i] += signal.convolve(self.h[b, i], self.e[b, o], 'valid')

    def __call__(self, x):
        return self.forward(x)


class MaxPool2d(Layer):
    def __init__(self, kernel_size, stride=(1, 1), padding=0):
        self.in_shape = None
        self.p = None
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

    @staticmethod
    def max(x):
        return np.max(x), np.argmax(x)

    def forward(self, x):
        self.in_shape = (x.shape[-2], x.shape[-1])
        r = np.zeros((x.shape[0], x.shape[1],
                      (x.shape[-2] - self.kernel_size[0] + self.stride[1]) // self.stride[1],
                      (x.shape[-1] - self.kernel_size[1] + self.stride[0]) // self.stride[0]))
        p = r.copy()
        for b in np.arange(r.shape[0]):
            for t in np.arange(r.shape[1]):
                for i in np.arange(0, r.shape[2]):
                    for j in np.arange(0, r.shape[3]):
                        r[b, t, i, j], p[b, t, i, j] = self.max(
                            x[b, t, i * self.stride[1]: i * self.stride[1] + self.kernel_size[0],
                            j * self.stride[0]: j * self.stride[0] + self.kernel_size[1]])
        self.p = p
        return r

    def backward(self, e):
        r = np.zeros((e.shape[0], e.shape[1],
                      e.shape[-2] * self.stride[1] + self.kernel_size[0] - self.stride[1],
                      e.shape[-1] * self.stride[0] + self.kernel_size[1] - self.stride[0]))
        for b in np.arange(0, e.shape[0]):
            for t in np.arange(0, e.shape[1]):
                for i in np.arange(0, e.shape[2]):
                    for j in np.arange(0, e.shape[3]):
                        p = (int(self.p[b, t, i, j] // self.kernel_size[1]),
                             int(self.p[b, t, i, j] % self.kernel_size[1]))
                        r[b, t, i * self.stride[1] + p[0], j * self.stride[0] + p[1]] = e[b, t, i, j]
                r[b, t] = np.pad(r[b, t], ((0, self.in_shape[-2] - r[b, t].shape[0]),
                                           (0, self.in_shape[-1] - r[b, t].shape[1])))
        return r

    def __call__(self, x):
        return self.forward(x)


class Flatten(Layer):
    def __init__(self, start_dim=1, end_dim=-1):
        self.s = None
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x):
        self.s = x.shape
        x = np.reshape(x, (*self.s[:self.start_dim], -1, *self.s[self.end_dim:][1:]))
        return x

    def backward(self, e):
        return e.reshape(self.s)

    def __call__(self, x):
        return self.forward(x)


class Relu(Layer):
    def __init__(self):
        self.h = None

    def forward(self, x):
        self.h = x
        for b in np.arange(x.shape[0]):
            x[b, x[b] < 0] = 0
        return x

    def backward(self, e):
        for b in np.arange(e.shape[0]):
            self.h[b, self.h[b] > 0] = 1
            self.h[b, self.h[b] < 0] = 0
            e[b] *= self.h[b]
        return e

    def __call__(self, x):
        return self.forward(x)


class Sigmod(Layer):
    def __init__(self):
        self.o = None

    def forward(self, x):
        self.o = np.empty(x.shape)
        for b in np.arange(x.shape[0]):
            self.o[b] = scipy.special.expit(x[b])
        return self.o

    def backward(self, e):
        for b in np.arange(e.shape[0]):
            e[b] *= self.o[b] * (1 - self.o[b])
        return e

    def __call__(self, x):
        return self.forward(x)
