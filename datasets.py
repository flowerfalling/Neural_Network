# -*- coding: utf-8 -*-
# @Time    : 2023/1/13 11:30
# @Author  : 之落花--falling_flowers
# @File    : datasets.py
# @Software: PyCharm
import csv

import numpy as np


class MNIST:
    def __init__(self, train: bool=True, batch_size: int=1, shuffle: bool=False):
        self.batch_size = batch_size
        self.counter = -1
        print('data loading...', end='')
        self.data = np.loadtxt('mnist_train.csv' if train else 'mnist_test.csv', delimiter=',')
        if shuffle:
            np.random.shuffle(self.data)
        self.amount = (60000 if train else 10000) // self.batch_size
        self.data = self.data[:self.amount * self.batch_size].reshape(
            (self.amount, self.batch_size, 785))
        print('done')

    def __iter__(self):
        return self

    def __next__(self):
        self.counter += 1
        if self.counter == self.amount:
            raise StopIteration
        r = self.data[self.counter]
        return np.array(r[:, 0], dtype=np.int8), np.asfarray(r[:, 1:]).reshape((self.batch_size, 1, 28, 28)) / 256
