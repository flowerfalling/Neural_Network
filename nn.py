# -*- coding: utf-8 -*-
# @Time    : 2023/1/13 10:47
# @Author  : 之落花--falling_flowers
# @File    : nn.py
# @Software: PyCharm
from abc import ABCMeta, abstractmethod
from typing import Union

import numpy as np
import scipy
from numpy.lib import stride_tricks
# from scipy import signal


class Layer(metaclass=ABCMeta):
    @abstractmethod
    def forward(self, x: "Tensor") -> "Tensor":
        pass

    @abstractmethod
    def backward(self, e: np.ndarray, parameter: dict) -> np.ndarray:
        pass


class LearnLayer(Layer, metaclass=ABCMeta):
    @abstractmethod
    def parameters(self):
        pass


class Linear(LearnLayer):
    def __init__(
            self,
            input_size: int,
            output_size: int,
            init_mode: str = 'k'
    ) -> None:
        if init_mode == 'k':
            variance = np.sqrt(2 / input_size)
        elif init_mode == 'x':
            variance = np.sqrt(2 / (input_size + output_size))
        else:
            variance = 1
        self.w = np.random.randn(output_size, input_size) * variance
        self.b = np.random.randn(output_size)
        self.e = None
        self.dw = np.zeros_like(self.w)
        self.db = np.zeros_like(self.b)

    def forward(self, x: "Tensor") -> "Tensor":
        x.cache.append((self, {'h': x.tensor.copy()}))
        x.tensor = np.einsum('bi,oi->bo', x.tensor, self.w, optimize='greedy') + self.b
        return x

    def backward(self, e: np.ndarray, parameter: dict) -> np.ndarray:
        self.e = e
        self.dw += np.einsum('bo,bi->oi', self.e, parameter['h'], optimize='greedy')
        self.db += np.einsum('bo->o', self.e, optimize='greedy')
        e = np.einsum('bo,oi->bi', e, self.w, optimize='greedy')
        return e

    def __call__(self, x: "Tensor") -> "Tensor":
        return self.forward(x)

    def parameters(self) -> list[list, list]:
        return [[self.w, self.dw], [self.b, self.db]]


class Conv2d(LearnLayer):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, tuple[int, int]],
            stride: Union[int, tuple[int, int]] = (1, 1),
            padding: Union[int, tuple[int, int]] = (0, 0),
            mode: str = 'CUSTOMIZE'
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = np.array(kernel_size) if isinstance(kernel_size, tuple) else np.array((kernel_size,) * 2)
        self.stride = np.array(stride) if isinstance(stride, tuple) else np.array((stride,) * 2)
        self.pad = np.array(padding) if isinstance(padding, tuple) else np.array((padding,) * 2)
        self.padding_mode = mode
        self.w = np.random.rand(
            out_channels, in_channels, *self.kernel_size) / (self.kernel_size[0] * self.kernel_size[1])
        self.dw = np.zeros_like(self.w)
        self.first_forward = True
        self.first_backward = True
        self.forward_path = None
        self.backward_path = None
        self.w_path = None
        self.require_grad = True

    def split(self, x: np.ndarray) -> np.ndarray:
        (B, C, H, W), (Kh, Kw), (Sh, Sw) = x.shape, self.kernel_size, self.stride
        shape = (B, C, (H - Kh) // Sh + 1, (W - Kw) // Sw + 1, Kh, Kw)
        s = x.strides
        strides = (s[0], s[1], s[2] * Sh, s[3] * Sw, *s[2:])
        return stride_tricks.as_strided(x, shape, strides, writeable=False)

    def padding(self, x: np.ndarray, forward: bool = True) -> np.ndarray:
        if forward:
            if self.padding_mode == 'VALID':
                return x
            elif self.padding_mode == 'SAME' and np.sum(self.pad):
                self.pad = self.kernel_size // 2
                self.padding_mode = 'CUSTOMIZE'
            return np.pad(x, ((0, 0), (0, 0), (self.pad[0],) * 2, (self.pad[1],) * 2))
        else:
            if self.padding_mode == 'CUSTOMIZE':
                return np.pad(x, ((0, 0), (0, 0), (self.kernel_size[0] - 1,) * 2, (self.kernel_size[1] - 1,) * 2))
            else:
                p = self.pad if self.padding_mode == 'SAME' else self.kernel_size - 1
                return np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))

    # noinspection PyTypeChecker
    def forward(self, x: "Tensor") -> "Tensor":
        x.tensor = self.padding(x.tensor)
        # h = x.tensor.copy()
        # x.tensor = signal.fftconvolve(x.tensor[:, None], np.flip(self.w)[None], 'valid')[:, :, 0]
        # x.cache.append((self, {'h': h, 'original_shape': x.tensor.shape}))
        # x.tensor = x.tensor[:, :, ::self.stride[0], ::self.stride[1]]
        shape = (*(np.array(x.tensor.shape[2:]) - self.kernel_size + 1),)
        r = self.split(x.tensor)
        x.cache.append((self, {'r': r, 's': shape}))
        if self.first_forward:
            self.first_forward = False
            self.w_path = np.einsum_path('bcij...,o...->boij', r, self.w, optimize='greedy')[0]
        x.tensor = np.einsum('bcij...,o...->boij', r, self.w, optimize=self.w_path)
        return x

    # noinspection PyTypeChecker
    def backward(self, e: np.ndarray, parameter: dict) -> np.ndarray:
        if self.require_grad:
            if self.first_backward:
                self.w_path = np.einsum_path('bojk,bijkcd->oicd', e, parameter['r'], optimize='greedy')[0]
            self.dw += np.einsum('bojk,bijkcd->oicd', e, parameter['r'], optimize=self.w_path)
        if np.sum(self.stride):
            temp = np.zeros((*e.shape[:2], *(parameter['s'])))
            temp[:, :, ::self.stride[0], ::self.stride[1]] = e
            e = temp
        e = self.split(self.padding(e, False))
        if self.first_backward:
            self.first_backward = False
            self.backward_path = np.einsum_path('bojkcd,oicd->bijk', e, self.w[:, :, ::-1, ::-1], optimize='greedy')[0]
        e = np.einsum('bojkcd,oicd->bijk', e, self.w[:, :, ::-1, ::-1], optimize=self.backward_path)
        # self.dw += signal.fftconvolve(parameter['h'][:, None], np.flip(e[:, :, None]), 'valid')[0]
        # e = np.pad(e, (*((0,) * 2,) * 2, *((self.kernel_size[0] - 1, self.kernel_size[1] - 1),) * 2))
        # e = signal.fftconvolve(e[:, :, None], np.flip(self.w[None, :, :, ::-1]), 'valid')[:, 0]
        # if self.padding:
        #     e = e[:, :, self.padding: -self.padding, self.padding: -self.padding]
        return e

    def __call__(self, x: "Tensor") -> "Tensor":
        return self.forward(x)

    def parameters(self):
        return [[self.w, self.dw]]


class MaxPool2d(Layer):
    def __init__(self, kernel_size: int or tuple[int, int],
                 stride: int or tuple[int, int] = (1, 1), padding=0):
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 2
        self.stride = stride if isinstance(stride, tuple) else (stride,) * 2
        self.padding = padding

    def forward(self, x: "Tensor") -> "Tensor":
        x.tensor = np.pad(x.tensor, (*((0,) * 2,) * 2, *((self.padding,) * 2,) * 2))
        s = x.tensor.shape
        r = stride_tricks.sliding_window_view(
            x.tensor, self.stride, axis=(-2, -1))[:, :, ::self.stride[0], ::self.stride[1]]
        x.tensor = np.max(r, axis=(-2, -1))
        i = np.argmax(r.reshape(*r.shape[:-2], -1), axis=-1)
        x.cache.append((self, {'ih': i // self.stride[1], 'iw': i % self.stride[1], 's': s}))
        return x

    def backward(self, e: np.ndarray, parameter: dict) -> np.ndarray:
        r = np.zeros(parameter['s'])
        s = e.shape
        for b in np.arange(0, s[0]):
            for t in np.arange(0, s[1]):
                for i in np.arange(0, s[2]):
                    for j in np.arange(0, s[3]):
                        r[b, t,
                        i * self.stride[0] + parameter['ih'][b, t, i, j],
                        j * self.stride[1] + parameter['iw'][b, t, i, j]] = e[b, t, i, j]
        return r

    def __call__(self, x: "Tensor") -> "Tensor":
        return self.forward(x)


class MaxPool(Layer):
    def __init__(self, size):
        self.size = size

    def forward(self, x: "Tensor") -> "Tensor":
        s = x.tensor.shape
        size = self.size
        out = x.tensor.reshape((s[0], s[1], s[2] // size, size, s[3] // size, size))
        out = out.max(axis=(3, 5))
        i = out.repeat(self.size, axis=-2).repeat(self.size, axis=-1) != x.tensor
        x.tensor = out
        x.cache.append((self, {'i': i}))
        return x

    def backward(self, e: np.ndarray, parameter: dict) -> np.ndarray:
        e = e.repeat(self.size, axis=-2).repeat(self.size, axis=-1)
        e[parameter['i']] = 0
        return e

    def __call__(self, x: "Tensor") -> "Tensor":
        return self.forward(x)


class BatchNorm1d(LearnLayer):
    def __init__(self, momentum: float = 0) -> None:
        self.gamma = 1
        self.beta = 0
        self.dgamma = 0
        self.dbeta = 0
        self.eps = 1e-8
        self.running_mean = 0
        self.running_var = 0
        self.momentum = momentum

    def forward(self, x: "Tensor", mode: str = 'train') -> "Tensor":
        if mode == 'train':
            mean = np.mean(x.tensor, axis=0, keepdims=True)
            var = np.var(x.tensor, axis=0, keepdims=True)
            sqrtvar = np.sqrt(var + self.eps)
            xnorm = (x.tensor - mean) / sqrtvar
            x.tensor = xnorm * self.gamma + self.beta
            x.cache.append((self, {'var': var, 'mean': mean, 'xnorm': xnorm, 'x': x.tensor.copy()}))
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            xnorm = (x.tensor - self.running_mean) / np.sqrt(self.running_var + self.eps)
            x.tensor = xnorm * self.gamma + self.beta
        return x

    def backward(self, e: np.ndarray, parameter: dict) -> np.ndarray:
        self.dbeta += np.sum(e, axis=0)
        self.dgamma += np.sum(e * parameter['xnorm'], axis=0)
        size = e.shape[1]
        dxnorm = e * self.gamma
        dvar = np.sum(dxnorm * (parameter['x'] - parameter['mean']) * (-0.5) *
                      np.power(parameter['var'] + self.eps, -1.5), axis=0)
        dmean = np.sum(dxnorm * (-1) * np.power(parameter['var'] + self.eps, -0.5), axis=0)
        dmean += dvar * np.sum(-2 * (parameter['x'] - parameter['mean']), axis=0) / size
        e = dxnorm * np.power(parameter['var'] + self.eps, -0.5) + dvar * 2 * (
                parameter['x'] - parameter['mean']) / size + dmean / size
        return e

    def __call__(self, x: "Tensor", mode: str = 'train') -> "Tensor":
        return self.forward(x, mode)

    def parameters(self) -> list[list, list]:
        return [[self.gamma, self.dgamma], [self.beta, self.dbeta]]


class Flatten(Layer):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: "Tensor") -> "Tensor":
        s = x.tensor.shape
        x.cache.append((self, {'s': s}))
        x.tensor = np.reshape(x.tensor, (*s[:self.start_dim], -1, *s[self.end_dim:][1:]))
        return x

    def backward(self, e: np.ndarray, parameter: dict) -> np.ndarray:
        return e.reshape(parameter['s'])

    def __call__(self, x: "Tensor") -> "Tensor":
        return self.forward(x)


class Dropout(Layer):
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def forward(self, x: "Tensor", state: str = 'train') -> "Tensor":
        if state == 'train':
            d = np.random.rand(*x.tensor.shape) < self.p
            x.tensor *= d / self.p
        return x

    def backward(self, e: np.ndarray, parameter: dict) -> np.ndarray:
        return e

    def __call__(self, x: "Tensor", state: str = 'train') -> "Tensor":
        return self.forward(x, state)


class Relu(Layer):
    def forward(self, x: "Tensor") -> "Tensor":
        x.cache.append((self, {'h': x.tensor.copy()}))
        x.tensor[x.tensor < 0] = 0
        return x

    def backward(self, e: np.ndarray, parameter: dict) -> np.ndarray:
        e[parameter['h'] < 0] = 0
        return e

    def __call__(self, x: "Tensor") -> "Tensor":
        return self.forward(x)


class Sigmod(Layer):
    def forward(self, x: "Tensor") -> "Tensor":
        x.tensor = scipy.special.expit(x.tensor)
        x.cache.append((self, {'o': x.tensor.copy()}))
        return x

    def backward(self, e: np.ndarray, parameter: dict) -> np.ndarray:
        return e * parameter['o'] * (1 - parameter['o'])

    def __call__(self, x: "Tensor") -> "Tensor":
        return self.forward(x)


class Tanh(Layer):
    def forward(self, x: "Tensor") -> "Tensor":
        x.tensor = np.tanh(x.tensor)
        x.cache.append({'o': x.tensor.copy()})
        return x

    def backward(self, e: np.ndarray, parameter: dict) -> np.ndarray:
        return e * (1 - parameter['o'] ** 2)

    def __call__(self, x: "Tensor") -> "Tensor":
        return self.forward(x)


class Softmax(Layer):
    def forward(self, x: "Tensor") -> "Tensor":
        pass

    def backward(self, e: np.ndarray, parameter: dict) -> np.ndarray:
        pass


class Start(Layer):
    def forward(self, x: object) -> None:
        pass

    def backward(self, e: object, parameter: object) -> None:
        pass


class Tensor:
    def __init__(self, tensor: np.ndarray, cache: Start = (Start(), {})) -> None:
        self.tensor: np.ndarray = tensor
        self.cache: list = [cache]

    def __call__(self) -> np.ndarray:
        return self.tensor

    def __str__(self) -> str:
        return self.tensor.__str__()

    def __add__(self, other) -> np.ndarray:
        if isinstance(other, Tensor):
            other.cache.extend(self.cache)
        return self.tensor + other.tensor
