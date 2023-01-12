# -*- coding: utf-8 -*-
# @Time    : 2023/1/6 16:36
# @Author  : 之落花--falling_flowers
# @File    : cnn_mnist.py
# @Software: PyCharm
import csv
from abc import ABCMeta, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import signal


class MNIST:
    def __init__(self, train=True, batch_size=1):
        self.batch_size = batch_size
        if train:
            with open('mnist_train.csv') as train_file:
                self.data = csv.reader(train_file)
                self.data = list(self.data)
                self.amount = 60000
        else:
            with open('mnist_test.csv') as test_file:
                self.data = csv.reader(test_file)
                self.data = list(self.data)
                self.amount = 10000

        self.counter = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.counter += 1
        if self.counter >= self.amount:
            raise StopIteration
        r = self.data[self.counter]
        return int(r[0]), np.asfarray(r[1:]).reshape((self.batch_size, 1, 28, 28)) / 256


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
    def __init__(self, lr, in_channels, out_channels, kernel_size, stride=1, padding=0):
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
                r[b, t] = signal.convolve(np.pad(e[b], ((0, 0), (2, 2), (2, 2))), self.w[:, t, ::-1], 'valid')
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
                        p = (int(self.p[i, j] // self.kernel_size[1]), int(self.p[i, j] % self.kernel_size[1]))
                        r[b, t, i * self.stride[1] + p[0], j * self.stride[0] + p[1]] = e[i, j]
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
        x = x.reshape(*(self.s[:self.start_dim], ), -1, *(self.s[self.end_dim:][1:], ))
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


class Net(Layer):
    def __init__(self, lr):
        self.lr = lr
        self.loss = []
        self.fc1 = Linear(self.lr, 784, 200)
        self.fc2 = Linear(self.lr, 200, 100)
        self.fc3 = Linear(self.lr, 100, 10)
        self.sigmod1 = Sigmod()
        self.sigmod2 = Sigmod()
        self.sigmod3 = Sigmod()

    def forward(self, x):
        x = self.sigmod1(self.fc1(x))
        x = self.sigmod2(self.fc2(x))
        x = self.sigmod3(self.fc3(x))
        return x

    def backward(self, e):
        pass

    def train(self, x, t):
        o = self(x.reshape(1, 784))
        e = t - o
        e = self.sigmod3.backward(e)
        e = self.fc3.backward(e)
        e = self.sigmod2.backward(e)
        e = self.fc2.backward(e)
        e = self.sigmod1.backward(e)
        e = self.fc1.backward(e)
        self.fc1.step()
        self.fc2.step()
        self.fc3.step()
        self.loss.append(e.sum() ** 2)

    def __call__(self, x):
        return self.forward(x)


class CNet(Layer):
    def __init__(self, lr):
        self.lr = lr
        self.loss = []
        self.conv1 = Conv2d(self.lr, 1, 10, (3, 3))
        self.conv2 = Conv2d(self.lr, 10, 16, (4, 4))
        self.pool1 = MaxPool2d((2, 2), (2, 2))
        self.pool2 = MaxPool2d((2, 2), (2, 2))
        self.flatten = Flatten()
        self.fc1 = Linear(self.lr, 400, 100)
        self.fc2 = Linear(self.lr, 100, 10)
        self.sigmod1 = Sigmod()
        self.sigmod2 = Sigmod()

    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.flatten(x)
        x = self.sigmod1(self.fc1(x))
        x = self.sigmod2(self.fc1(x))
        return x

    def backward(self, e):
        pass

    def train(self, x, t):
        o = self(x)
        e = t - o
        e = self.sigmod2.backward(e)
        e = self.fc2.backward(e)
        e = self.sigmod1.backward(e)
        e = self.fc1.backward(e)
        e = self.flatten.backward(e)
        e = self.pool2.backward(e)
        e = self.conv2.backward(e)
        e = self.pool1.backward(e)
        e = self.conv1.backward(e)
        self.conv1.step()
        self.conv2.step()
        self.fc1.step()
        self.fc2.step()
        self.loss.append(e.sum() ** 2)

    def __call__(self, x):
        return self.forward(x)


def main():
    # np.random.seed(1)
    # net = Net(0.01)
    net = CNet(0.01)
    ts = np.eye(10)[:, None]
    for label, data in MNIST(True):
        net.train(data, ts[label])
    plt.plot(net.loss)
    plt.show()
    # np.save('./pth/net_fc1w.npy', net.fc1.w)
    # np.save('./pth/net_fc2w.npy', net.fc2.w)
    # np.save('./pth/net_fc3w.npy', net.fc3.w)
    # np.save('./pth/net_fc1b.npy', net.fc1.b)
    # np.save('./pth/net_fc2b.npy', net.fc2.b)
    # np.save('./pth/net_fc3b.npy', net.fc3.b)
    # net.fc1.w = np.load('./pth/net_fc1w.npy')
    # net.fc2.w = np.load('./pth/net_fc2w.npy')
    # net.fc3.w = np.load('./pth/net_fc3w.npy')
    # net.fc1.b = np.load('./pth/net_fc1b.npy')
    # net.fc2.b = np.load('./pth/net_fc2b.npy')
    # net.fc3.b = np.load('./pth/net_fc3b.npy')
    c = 0
    for label, data in MNIST(False):
        if np.argmax(net(data.reshape(1, 784))) == label:
            c += 1
    print(c)
    pass


if __name__ == '__main__':
    main()
