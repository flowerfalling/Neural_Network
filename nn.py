# -*- coding: utf-8 -*-
# @Time    : 2023/1/13 10:47
# @Author  : 之落花--falling_flowers
# @File    : nn.py
# @Software: PyCharm
import numpy as np
import scipy
from abc import ABCMeta, abstractmethod
from scipy import signal


class Layer(metaclass=ABCMeta):
    def __init__(self):
        self.from_layer = Cache()

    def navigate(self, x):
        if isinstance(x[1], Layer):
            self.from_layer.push(x[1])
            return x[0]
        return x

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, e):
        pass


class LearnLayer(Layer, metaclass=ABCMeta):
    def __init__(self, lr):
        super().__init__()
        self.lr = lr

    @abstractmethod
    def step(self):
        pass


class Linear(LearnLayer):
    def __init__(self, lr, input_size, output_size):
        super().__init__(lr)
        self.w = np.random.rand(output_size, input_size) - 0.5
        self.b = np.random.rand(output_size) - 0.5
        self.h = None
        self.e = None

    def forward(self, x):
        x = super().navigate(x)
        self.h = x
        r = np.empty((x.shape[0], *self.b.shape))
        for b in np.arange(x.shape[0]):
            r[b] = np.dot(self.w, x[b]) + self.b
        return r, self

    def backward(self, e):
        from_layer = self.from_layer.pop()
        if from_layer is None:
            return
        self.e = e
        r = np.empty(self.h.shape)
        for b in np.arange(e.shape[0]):
            r[b] = np.dot(self.w.T, e[b])
        from_layer.backward(r)

    def step(self):
        for b in np.arange(self.e.shape[0]):
            self.w += np.dot(self.e[b, None].T, self.h[b, None]) * self.lr
            self.b += self.e[b] * self.lr

    def __call__(self, x):
        return self.forward(x)


class Conv2d(LearnLayer):
    def __init__(self, lr, in_channels, out_channels, kernel_size, stride=(1, 1), padding=0):
        super().__init__(lr)
        self.h = None
        self.e = None
        self.stride = stride
        self.padding = padding
        self.w = np.random.rand(out_channels, in_channels, kernel_size[0], kernel_size[1]) - 0.5

    def forward(self, x):
        x = super().navigate(x)
        self.h = x
        r = [[0 for _ in range(self.w.shape[0])] for _ in range(x.shape[0])]
        for b in range(x.shape[0]):
            for t in range(self.w.shape[0]):
                r[b][t] = signal.convolve(x[b], self.w[t], 'valid')
        return np.array(r)[:, :, 0], self

    def backward(self, e):
        from_layer = self.from_layer.pop()
        if from_layer is None:
            return
        self.e = e
        r = np.empty(self.h.shape)
        for b in np.arange(e.shape[0]):
            for t in np.arange(self.w.shape[1]):
                pad = (self.w.shape[-2] - self.stride[0], self.w.shape[-1] - self.stride[1])
                r[b, t] = signal.convolve(np.pad(e[b], ((0, 0), (pad[0], pad[0]), (pad[1], pad[1]))),
                                          self.w[:, t, ::-1], 'valid')
        from_layer.backward(r)

    def step(self):
        for b in np.arange(self.e.shape[0]):
            for o in np.arange(self.w.shape[0]):
                for i in np.arange(self.w.shape[1]):
                    self.w[o, i] += signal.convolve(self.h[b, i], self.e[b, o], 'valid')

    def __call__(self, x):
        return self.forward(x)


class MaxPool2d(Layer):
    def __init__(self, kernel_size, stride=(1, 1), padding=0):
        super().__init__()
        self.in_shape = Cache()
        self.p = Cache()
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

    @staticmethod
    def max(x):
        return np.max(x), np.argmax(x)

    def forward(self, x):
        x = super().navigate(x)
        self.in_shape.push((x.shape[-2], x.shape[-1]))
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
        self.p.push(p)
        return r, self

    def backward(self, e):
        from_layer = self.from_layer.pop()
        if from_layer is None:
            return
        r = np.zeros((e.shape[0], e.shape[1],
                      e.shape[-2] * self.stride[1] + self.kernel_size[0] - self.stride[1],
                      e.shape[-1] * self.stride[0] + self.kernel_size[1] - self.stride[0]))
        ps = self.p.pop()
        in_shape = self.in_shape.pop()
        for b in np.arange(0, e.shape[0]):
            for t in np.arange(0, e.shape[1]):
                for i in np.arange(0, e.shape[2]):
                    for j in np.arange(0, e.shape[3]):
                        p = (int(ps[b, t, i, j] // self.kernel_size[1]),
                             int(ps[b, t, i, j] % self.kernel_size[1]))
                        r[b, t, i * self.stride[1] + p[0], j * self.stride[0] + p[1]] = e[b, t, i, j]
                r[b, t] = np.pad(r[b, t], ((0, in_shape[-2] - r[b, t].shape[0]),
                                           (0, in_shape[-1] - r[b, t].shape[1])))
        from_layer.backward(r)

    def __call__(self, x):
        return self.forward(x)


class Flatten(Layer):
    def __init__(self, start_dim=1, end_dim=-1):
        super().__init__()
        self.s = Cache()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x):
        x = super().navigate(x)
        s = x.shape
        self.s.push(s)
        x = np.reshape(x, (*s[:self.start_dim], -1, *s[self.end_dim:][1:]))
        return x, self

    def backward(self, e):
        from_layer = self.from_layer.pop()
        if from_layer is None:
            return
        from_layer.backward(e.reshape(self.s.pop()))

    def __call__(self, x):
        return self.forward(x)


class Relu(Layer):
    def __init__(self):
        super().__init__()
        self.h = Cache()

    def forward(self, x):
        x = super().navigate(x)
        self.h.push(x)
        x[x < 0] = 0
        return x, self

    def backward(self, e):
        h = self.h.pop()
        h[h > 0] = 1
        h[h < 0] = 0
        return e * h

    def __call__(self, x):
        return self.forward(x)


class Sigmod(Layer):
    def __init__(self):
        super().__init__()
        self.o = Cache()

    def forward(self, x):
        x = super().navigate(x)
        self.o.push(scipy.special.expit(x))
        return self.o.peek(), self

    def backward(self, e):
        o = self.o.pop()
        return e * o * (1 - o)

    def __call__(self, x):
        return self.forward(x)


class Cache:
    def __init__(self):
        self.__items = []

    def __bool__(self):
        return self.__items != []

    def __iter__(self):
        return self.__items[::-1]

    def push(self, item):
        self.__items.append(item)

    def pop(self):
        if not self:
            return None
        return self.__items.pop()

    def peek(self):
        if not self:
            return None
        return self.__items[-1]

    def clear(self):
        self.__items.clear()
