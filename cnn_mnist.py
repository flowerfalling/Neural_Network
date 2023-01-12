# -*- coding: utf-8 -*-
# @Time    : 2023/1/6 16:36
# @Author  : 之落花--falling_flowers
# @File    : cnn_mnist.py
# @Software: PyCharm
import csv
import numpy as np
from abc import ABCMeta, abstractmethod
import scipy
import matplotlib.pyplot as plt
from scipy import signal


class MNIST:
    def __init__(self, train=True):
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
        return int(r[0]), np.asfarray(r[1:]).reshape(28, 28) / 256


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
        self.b = np.random.rand(output_size, 1) - 0.5

    def forward(self, x):
        self.h = x
        return np.dot(self.w, x) + self.b

    def backward(self, e):
        self.e = e
        return np.dot(self.w.T, e)

    def step(self):
        self.w += np.dot(self.e, self.h.T) * self.lr
        self.b += self.e * self.lr

    def __call__(self, x):
        return self.forward(x)


class Conv2d(Layer):
    def __init__(self, lr, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.lr = lr
        self.h = None
        self.e = None
        self.stride = stride
        self.padding = padding
        self.w = np.random.rand(out_channels, in_channels, kernel_size, kernel_size) - 0.5

    def forward(self, x):
        self.h = x
        return signal.convolve2d(x, self.w, 'valid')

    def backward(self, e):
        self.e = e
        return signal.convolve2d(self.h, e[:, ::-1], 'valid')

    def step(self):
        self.w += signal.convolve2d(self.h, self.e, 'valid')


class MaxPool2d(Layer):
    def __init__(self, kernel_size, stride=1, padding=0):
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

    def forward(self, x):
        pass

    def backward(self, e):
        pass


class Relu(Layer):
    def __init__(self):
        self.h = None

    def forward(self, x):
        self.h = x
        x[x < 0] = 0
        return x

    def backward(self, e):
        self.h[self.h > 0] = 1
        self.h[self.h < 0] = 0
        return e * self.h

    def __call__(self, x):
        return self.forward(x)


class Sigmod(Layer):
    def __init__(self):
        self.o = None

    def forward(self, x):
        self.o = scipy.special.expit(x)
        return self.o

    def backward(self, e):
        return e * self.o * (1 - self.o)

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
        x = self.sigmod1(self.fc1(x.T))
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
        self.conv1 = Conv2d(1, 10, 3)

    def forward(self, x):
        pass

    def backward(self, e):
        pass


def main():
    np.random.seed(1)
    net = Net(0.01)
    ts = np.eye(10).reshape(10, 10, 1)
    for label, data in MNIST(True):
        net.train(data, ts[label])
    c = 0
    for label, data in MNIST(False):
        if np.argmax(net(data)) == label:
            c += 1
    print(c)
    pass


if __name__ == '__main__':
    main()
