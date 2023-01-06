# -*- coding: utf-8 -*-
# @Time    : 2023/1/6 16:36
# @Author  : 之落花--falling_flowers
# @File    : cnn_mnist.py
# @Software: PyCharm
import csv
import numpy as np
from abc import ABCMeta, abstractmethod


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
    @staticmethod
    def relu(x):
        return np.max(0, x)

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def update(self):
        pass


class Linear(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)

    def forward(self, x):
        return self.relu(np.dot(self.weights, x))

    def update(self):
        pass


def main():
    trainset = MNIST()
    pass


if __name__ == '__main__':
    main()
