# -*- coding: utf-8 -*-
# @Time    : 2023/1/6 16:36
# @Author  : 之落花--falling_flowers
# @File    : nndemo_mnist.py
# @Software: PyCharm
import numpy as np
import time
from matplotlib import pyplot as plt

import datasets
import nn


class CNet(nn.LearnLayer):
    def __init__(self, lr):
        super().__init__(lr)
        self.loss = []
        self.conv1 = nn.Conv2d(lr, 1, 10, (3, 3))
        self.conv2 = nn.Conv2d(lr, 10, 16, (4, 4))
        self.pool = nn.MaxPool2d((2, 2), (2, 2))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(lr, 400, 120)
        self.fc2 = nn.Linear(lr, 120, 84)
        self.fc3 = nn.Linear(lr, 84, 10)
        self.sigmod = nn.Sigmod()
        self.relu = nn.Relu()

    def forward(self, x: np.ndarray):
        x = self.pool(self.relu(self.conv1((x, nn.Start()))))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmod(self.fc3(x))
        self.from_layer.push(x[1])
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

    def __call__(self, x):
        return self.forward(x)


class Net(nn.LearnLayer):
    def __init__(self, lr):
        super().__init__(lr)
        self.lr = lr
        self.loss = []
        self.fc1 = nn.Linear(self.lr, 784, 200)
        self.fc2 = nn.Linear(self.lr, 200, 100)
        self.fc3 = nn.Linear(self.lr, 100, 10, 'x')
        self.flatten = nn.Flatten()
        self.sigmod = nn.Sigmod()
        self.relu = nn.Relu()
        self.tanh = nn.Tanh()

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = self.flatten((x, nn.Start()))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmod(self.fc3(x))
        self.from_layer.push(x[1])
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

    def __call__(self, x):
        return self.forward(x)


def main():
    np.random.seed(1)
    trainset = datasets.MNIST(True, 10, True)
    t = time.time()
    net = Net(0.01)
    ts = np.eye(10)
    for label, data in trainset:
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
