# -*- coding: utf-8 -*-
# @Time    : 2023/1/6 16:36
# @Author  : 之落花--falling_flowers
# @File    : nndemo_mnist.py
# @Software: PyCharm
import time

import numpy as np
from matplotlib import pyplot as plt

import demo
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


class CNet(nn.LearnLayer):
    def __init__(self, lr):
        super().__init__(lr)
        self.loss = []
        self.conv1 = demo.Conv2d(lr, 1, 10, (3, 3))
        self.conv2 = demo.Conv2d(lr, 10, 16, (4, 4))
        self.pool = nn.MaxPool2d((2, 2), (2, 2))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(lr, 400, 100)
        self.fc2 = nn.Linear(lr, 100, 10)
        self.sigmod = nn.Sigmod()

    def forward(self, x):
        x = self.sigmod(self.pool(self.conv1(x)))
        x = self.sigmod(self.pool(self.conv2(x)))
        x = self.flatten(x)
        x = self.sigmod(self.fc1(x))
        x = self.sigmod(self.fc2(x))
        self.from_layer.push(self.sigmod)
        return x[0]

    def backward(self, e):
        from_layer = self.from_layer.pop()
        if from_layer is None:
            return
        from_layer.backward(e)

    def step(self):
        for layer in self.__dict__.values():
            if isinstance(layer, nn.LearnLayer):
                layer.step()

    def train(self, x, t):
        o = self(x)
        e = t - o
        self.loss.append((e ** 2).sum())
        self.backward(e)
        self.step()
        # e = self.sigmod.backward(e)
        # e = self.fc2.backward(e)
        # e = self.sigmod.backward(e)
        # e = self.fc1.backward(e)
        # e = self.flatten.backward(e)
        # e = self.sigmod.backward(e)
        # e = self.pool.backward(e)
        # e = self.conv2.backward(e)
        # e = self.sigmod.backward(e)
        # e = self.pool.backward(e)
        # self.conv1.backward(e)
        # self.conv1.step()
        # self.conv2.step()
        # self.fc1.step()
        # self.fc2.step()

    def __call__(self, x):
        return self.forward(x)


def main():
    # t = time.time()
    # np.random.seed(1)
    # # net = Net(0.01)
    net = CNet(0.01)
    ts = np.eye(10)
    for label, data in datasets.MNIST(True, 10):
        for _ in range(20):
            net.train(data, ts[label])
        break
    # print(time.time() - t)
    plt.plot(net.loss)
    plt.show()
    pass
    # t = time.time()
    # c = 0
    # for label, data in datasets.MNIST(False):
    #     if np.argmax(net(data)) == label:
    #         c += 1
    # print(time.time() - t)
    # print(c)


if __name__ == '__main__':
    main()
