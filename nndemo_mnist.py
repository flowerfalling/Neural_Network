# -*- coding: utf-8 -*-
# @Time    : 2023/1/6 16:36
# @Author  : 之落花--falling_flowers
# @File    : nndemo_mnist.py
# @Software: PyCharm
import time

import numpy as np
from matplotlib import pyplot as plt

import datasets
import nn
import optim


class Net(nn.LearnLayer):
    def __init__(self):
        self.state = 'train'
        self.loss = []
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 10, 'x')
        self.flatten = nn.Flatten()
        self.relu = nn.Relu()
        self.softmax = nn.Softmax()

    def forward(self, x: np.ndarray) -> nn.Tensor:
        x = nn.Tensor(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x

    def backward(self, e, parameter: dict):
        pass

    def __call__(self, x):
        return self.forward(x)

    def parameters(self) -> list:
        parameter = []
        for i in self.__dict__.values():
            if isinstance(i, nn.LearnLayer):
                parameter.extend(i.parameters())
        return parameter

    def load(self):
        net_parameter = np.load(r'.\pth\net.npz')
        self.fc1.w = net_parameter['fc1_w']
        self.fc1.b = net_parameter['fc1_b']
        self.fc2.w = net_parameter['fc2_w']
        self.fc2.b = net_parameter['fc2_b']
        self.fc3.w = net_parameter['fc3_w']
        self.fc3.b = net_parameter['fc3_b']
        self.loss = list(net_parameter['loss'])

    def save(self):
        np.savez(
            r'.\pth\net.npz',
            fc1_w=self.fc1.w,
            fc2_w=self.fc2.w,
            fc3_w=self.fc3.w,
            fc1_b=self.fc1.b,
            fc2_b=self.fc2.b,
            fc3_b=self.fc3.b,
            loss=np.array(self.loss)
        )


def train(epoches):
    ts = np.eye(10)
    net = Net()
    # net.load()
    optimizer = optim.Adam(net.parameters(), 0.001, 0.9, 0.9)
    loss_fn = optim.CrossEntropyLoss()
    batch_size = 10
    trainset = datasets.MNIST(True, batch_size, True)
    testset = datasets.MNIST(False)
    for epoch in range(epoches):
        plt.title('net.loss')
        plt.xlabel('batches')
        plt.ylabel('loss')
        net.state = 'train'
        t = time.time()
        for label, data in trainset:
            target = ts[label]
            out = net(data)
            e = target - out.tensor
            net.loss.append(loss_fn(out, target))
            for i in out.cache[::-1]:
                e = i[0].backward(e, i[1])
            optimizer.step()
        print('epoch: {}    use time: {:.3f}s    '.format(epoch + 1, time.time() - t), end='')
        plt.scatter(np.arange(1, len(net.loss) + 1), np.array(net.loss) / batch_size, 3, marker='.')
        plt.show()
        # net.loss.clear()
        net.state = 'test'
        c = 0
        for label, data in testset:
            if np.argmax(net(data).tensor) == label:
                c += 1
        print(f'Correct rate: {c / 100}%')
    net.save()


def main():
    np.random.seed(2)
    epoch = 10
    train(epoch)


if __name__ == '__main__':
    main()
