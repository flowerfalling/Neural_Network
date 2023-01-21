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
import optim


# class CNet(nn.LearnLayer):
#     def __init__(self, lr):
#         super().__init__(lr)
#         self.loss = []
#         self.conv1 = nn.Conv2d(lr, 1, 10, (3, 3))
#         self.conv2 = nn.Conv2d(lr, 10, 16, (4, 4))
#         self.pool = nn.MaxPool2d((2, 2), (2, 2))
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(lr, 400, 120)
#         self.fc2 = nn.Linear(lr, 120, 84)
#         self.fc3 = nn.Linear(lr, 84, 10)
#         self.sigmod = nn.Sigmod()
#         self.relu = nn.Relu()
#
#     def forward(self, x: np.ndarray):
#         x = self.pool(self.relu(self.conv1((x, nn.Start()))))
#         x = self.pool(self.relu(self.conv2(x)))
#         x = self.flatten(x)
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))
#         x = self.sigmod(self.fc3(x))
#         self.from_layer.append(x[1])
#         return x[0]
#
#     def backward(self, e):
#         from_layer = self.from_layer.pop()
#         if from_layer is None:
#             return
#         from_layer.backward(e, )
#
#     def step(self):
#         for layer in self.__dict__.values():
#             if isinstance(layer, nn.LearnLayer):
#                 layer.step()
#
#     def train(self, x, t):
#         o = self(x)
#         e = t - o
#         self.loss.append((e ** 2).sum())
#         self.backward(e)
#         self.step()
#
#     def __call__(self, x):
#         return self.forward(x)


class Net(nn.LearnLayer):
    def __init__(self):
        self.state = 'train'
        self.loss = []
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 10, 'x')
        self.flatten = nn.Flatten()
        self.sigmod = nn.Sigmod()
        self.relu = nn.Relu()
        self.bn = nn.BatchNorm1d(0.9)
        self.dropout = nn.Dropout(0.8)

    def forward(self, x: np.ndarray) -> nn.Tensor:
        x = nn.Tensor(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmod(self.fc3(x))
        return x

    def backward(self, e, parameter: dict):
        pass

    def train(self, x: nn.Tensor, t: np.ndarray):
        o = self(x)
        e = t - o.tensor
        self.loss.append((e ** 2).sum())
        for i in o.cache[::-1]:
            e = i[0].backward(e, i[1])

    def __call__(self, x):
        return self.forward(x)

    def parameters(self) -> list:
        parameter = []
        for i in self.__dict__.values():
            if isinstance(i, nn.LearnLayer):
                parameter.extend(i.parameters())
        return parameter

    def load(self):
        self.fc1.w = np.load(r'.\pth\net\net-fc1-w.npy')
        self.fc1.b = np.load(r'.\pth\net\net-fc1-b.npy')
        self.fc2.w = np.load(r'.\pth\net\net-fc2-w.npy')
        self.fc2.b = np.load(r'.\pth\net\net-fc2-b.npy')
        self.fc3.w = np.load(r'.\pth\net\net-fc3-w.npy')
        self.fc3.b = np.load(r'.\pth\net\net-fc3-b.npy')
        self.loss = np.load(r'.\pth\net\net-loss.npy')


def main():
    np.random.seed(2)
    plt.title('net.loss')
    plt.xlabel('batches')
    plt.ylabel('loss')
    batch_size = 10
    net = Net()
    trainset = datasets.MNIST(True, batch_size, True)
    testset = datasets.MNIST(False)
    optimizer = optim.Adam(net.parameters(), 0.001, 0.9, 0.9)
    ts = np.eye(10)
    epoch = 10
    for e in range(epoch):
        net.state = 'train'
        t = time.time()
        for label, data in trainset:
            net.train(data, ts[label])
            optimizer.step()

        print('epoch: {}    use time: {:.3f}s'.format(e + 1, time.time() - t), end='')
        plt.scatter(np.arange(1, len(net.loss) + 1), np.array(net.loss) / batch_size, 3, marker='.')
        plt.show()
        net.loss.clear()
        net.state = 'test'

        c = 0
        for label, data in testset:
            if np.argmax(net(data).tensor) == label:
                c += 1
        print(f'    Correct rate: {c / 100}%')


if __name__ == '__main__':
    main()
