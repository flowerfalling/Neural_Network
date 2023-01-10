# -*- coding: utf-8 -*-
# @Time    : 2023/1/6 16:36
# @Author  : 之落花--falling_flowers
# @File    : cnn_mnist.py
# @Software: PyCharm
import csv
import numpy as np
from abc import ABCMeta, abstractmethod
import scipy


class MNIST:
    def __init__(self, train=True):
        if train:
            with open('mnist_train.csv') as train_file:
                self.data = csv.reader(train_file)
                self.data = list(self.data)
                self.amount = 5000
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
        if self.counter > self.amount:
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
    def __init__(self, input_size, output_size, lr):
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

    def step(self, e):
        self.w -= np.dot(e, self.h.T)
        self.b -= e

    def __call__(self, x):
        return self.forward(x)


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
        self.fc1 = Linear(784, 200, self.lr)
        self.fc2 = Linear(200, 100, self.lr)
        self.fc3 = Linear(100, 10, self.lr)
        self.relu1 = Relu()
        self.relu2 = Relu()
        self.sigmod = Sigmod()

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.sigmod(self.fc3(x))
        return x

    def backward(self, e):
        pass

    def train(self, x, t):
        o = self(x)
        e = (t - o) * self.lr
        e = self.sigmod.backward(e)
        self.fc3.step(e)
        e = self.fc3.backward(e)
        e = self.relu2.backward(e)
        self.fc2.step(e)
        e = self.fc2.backward(e)
        e = self.relu1.backward(e)
        self.fc1.step(e)
        pass

    def __call__(self, x):
        return self.forward(x)


def main():
    trainset = MNIST()
    net = Net(0.001)
    for label, data in trainset:
        net.train(data.reshape(784, 1), label)
    c = 0
    for label, data in MNIST():
        if np.argmax(net(data.reshape(784, 1))) == label:
            c += 1
    print(c)


if __name__ == '__main__':
    main()
