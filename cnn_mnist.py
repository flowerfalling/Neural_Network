# -*- coding: utf-8 -*-
# @Time    : 2023/1/6 16:36
# @Author  : 之落花--falling_flowers
# @File    : cnn_mnist.py
# @Software: PyCharm
import time

import numpy as np
from matplotlib import pyplot as plt

import nn
import datasets


class Net(nn.Layer):
    def __init__(self, lr):
        self.lr = lr
        self.loss = []
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.lr, 784, 200)
        self.fc2 = nn.Linear(self.lr, 200, 100)
        self.fc3 = nn.Linear(self.lr, 100, 10)
        self.sigmod1 = nn.Sigmod()
        self.sigmod2 = nn.Sigmod()
        self.sigmod3 = nn.Sigmod()

    def forward(self, x):
        x = self.flatten(x)
        x = self.sigmod1(self.fc1(x))
        x = self.sigmod2(self.fc2(x))
        x = self.sigmod3(self.fc3(x))
        return x

    def backward(self, e):
        pass

    def train(self, x, t):
        o = self(x)
        e = t - o
        self.loss.append((e ** 2).sum())
        e = self.sigmod3.backward(e)
        e = self.fc3.backward(e)
        e = self.sigmod2.backward(e)
        e = self.fc2.backward(e)
        e = self.sigmod1.backward(e)
        self.fc1.backward(e)
        self.fc1.step()
        self.fc2.step()
        self.fc3.step()

    def __call__(self, x):
        return self.forward(x)


class CNet(nn.Layer):
    def __init__(self, lr):
        self.lr = lr
        self.loss = []
        self.conv1 = nn.Conv2d(self.lr, 1, 10, (3, 3))
        self.conv2 = nn.Conv2d(self.lr, 10, 16, (4, 4))
        self.pool1 = nn.MaxPool2d((2, 2), (2, 2))
        self.pool2 = nn.MaxPool2d((2, 2), (2, 2))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.lr, 400, 100)
        self.fc2 = nn.Linear(self.lr, 100, 10)
        self.sigmod1 = nn.Sigmod()
        self.sigmod2 = nn.Sigmod()
        self.sigmod3 = nn.Sigmod()
        self.sigmod4 = nn.Sigmod()

    def forward(self, x):
        x = self.sigmod1(self.pool1(self.conv1(x)))
        x = self.sigmod2(self.pool2(self.conv2(x)))
        x = self.flatten(x)
        x = self.sigmod3(self.fc1(x))
        x = self.sigmod4(self.fc2(x))
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
    t = time.time()
    np.random.seed(1)
    net = Net(0.01)
    ts = np.eye(10)
    for label, data in datasets.MNIST(True, 10):
        net.train(data, ts[label])
    print(time.time() - t)
    plt.plot(net.loss)
    plt.show()
    t = time.time()
    c = 0
    for label, data in datasets.MNIST(False):
        if np.argmax(net(data)) == label:
            c += 1
    print(time.time() - t)
    print(c)


if __name__ == '__main__':
    main()
