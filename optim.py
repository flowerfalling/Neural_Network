# -*- coding: utf-8 -*-
# @Time    : 2023/1/22 1:11
# @Author  : 之落花--falling_flowers
# @File    : optim.py
# @Software: PyCharm
from abc import ABCMeta, abstractmethod
import numpy as np


class Optim(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, parameter: list, lr: float) -> None:
        self.parameter = parameter
        self.lr = lr

    @abstractmethod
    def step(self) -> None:
        pass


class GD(Optim):
    def __init__(self, parameter: list, lr: float) -> None:
        super().__init__(parameter, lr)

    def step(self) -> None:
        for i in self.parameter:
            i[0] += i[1] * self.lr
            i[1] *= 0


class MGD(Optim):
    def __init__(self, parameter: list, lr: float, beta: float = 0) -> None:
        super().__init__(parameter, lr)
        self.beta = beta
        for i in self.parameter:
            i.append(0)

    def step(self) -> None:
        for i in self.parameter:
            i[2] *= self.beta
            i[2] += (1 - self.beta) * i[1]
            i[0] += i[2] * self.lr
            i[1] *= 0


class RMSprop(Optim):
    def __init__(self, parameter: list, lr: float, beta: float = 0) -> None:
        super().__init__(parameter, lr)
        self.beta = beta
        self.eps = 1e-8
        for i in self.parameter:
            i.append(0)

    def step(self) -> None:
        for i in self.parameter:
            i[2] *= self.beta
            i[2] += (1 - self.beta) * i[1] ** 2
            i[0] += i[1] / np.sqrt(i[2] + self.eps) * self.lr
            i[1] *= 0


class Adam(Optim):
    def __init__(self, parameter: list, lr: float, beta1: float = 0, beta2: float = 0) -> None:
        super().__init__(parameter, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = 1e-8
        self.count = 1
        for i in self.parameter:
            i += [0, 0]

    def step(self) -> None:
        for i in self.parameter:
            i[2] *= self.beta1
            i[2] += (1 - self.beta1) * i[1]
            if self.count < 100:
                i[2] /= (1 - self.beta1 ** self.count)
                self.count += 1
            i[3] *= self.beta2
            i[3] += (1 - self.beta2) * i[1] ** 2
            i[0] += i[2] / np.sqrt(i[3] + self.eps) * self.lr
            i[1] *= 0
