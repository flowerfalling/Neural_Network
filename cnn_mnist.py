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
    def __init__(self, batch_size=1, train=True):
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
        if self.counter > self.amount:
            raise StopIteration
        r = self.data[self.counter]
        return int(r[0]), np.asfarray(r[1:]).reshape(28, 28) / 256


class Layer(metaclass=ABCMeta):
    def __init__(self, lr=0):
        self.lr = lr

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def update(self, e, h):
        pass


class Linear(Layer):
    def __init__(self, input_size, output_size, lr):
        super().__init__(lr)
        self.weights = np.random.randn(output_size, input_size)

    def forward(self, x):
        return np.dot(self.weights, x)

    def update(self, e, h):
        self.weights += self.lr * np.dot(e, h.T)
        e = np.dot(e, self.weights.T)
        return e

    def __call__(self, x):
        return self.forward(x)


class Relu(Layer):
    def forward(self, x):
        x[x < 0] = 0
        return x

    @staticmethod
    def reforward(x):
        x[x > 0] = 1
        x[x < 0] = 0
        return x

    def update(self, e, h):
        return e * self.reforward(h)

    def __call__(self, x):
        return self.forward(x)


class Sigmod(Layer):
    def forward(self, x):
        return scipy.special.expit(x)

    def update(self, e, o):
        return e * o * (1 - o)

    def __call__(self, x):
        self.forward(x)

class Sigmod(Layer):
    def forward(self, x):
        pass

    def update(self, e, h):
        pass


class Net(Layer):
    def __init__(self, lr):
        super().__init__(lr)
        self.fc1 = Linear(784, 200, self.lr)
        self.fc2 = Linear(200, 100, self.lr)
        self.fc3 = Linear(100, 10, self.lr)
        self.relu = Relu()
        self.sigmod = Sigmod()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        pass

    def update(self, e, h):
        pass


def main():
    pass


if __name__ == '__main__':
    main()
