# -*- coding: utf-8 -*-
# @Time    : 2023/1/13 10:47
# @Author  : 之落花--falling_flowers
# @File    : nn.py
# @Software: PyCharm
import numpy as np
import scipy
from abc import ABCMeta, abstractmethod
from scipy import signal


class Layer(metaclass=ABCMeta):
    def __init__(self):
        self.from_layer = Cache()

    def navigate(self, x: tuple[np.ndarray, "Layer"]) -> np.ndarray:
        self.from_layer.push(x[1])
        return x[0]

    @abstractmethod
    def forward(self, x: tuple[np.ndarray, "Layer"]) -> tuple[np.ndarray, "Layer"]:
        pass

    @abstractmethod
    def backward(self, e: np.ndarray) -> None:
        pass


class LearnLayer(Layer, metaclass=ABCMeta):
    def __init__(self, lr: float) -> None:
        super().__init__()
        self.lr = lr

    @abstractmethod
    def step(self) -> None:
        pass


class Linear(LearnLayer):
    def __init__(
            self, lr: float,
            input_size: int,
            output_size: int,
            init_mode: str = 'k',
    ) -> None:
        super().__init__(lr)
        if init_mode == 'k':
            variance = 2 / input_size ** 0.5
        elif init_mode == 'x':
            variance = 2 / (input_size + output_size)
        else:
            variance = 1
        self.w = np.random.randn(output_size, input_size) * variance
        self.b = np.random.randn(output_size)
        self.h = None
        self.e = None

    def forward(self, x: tuple[np.ndarray, Layer]) -> tuple[np.ndarray, Layer]:
        x = super().navigate(x)
        self.h = x
        r = np.einsum('bi,oi->bo', x, self.w, optimize=True) + self.b
        return r, self

    def backward(self, e: np.ndarray) -> None:
        self.e = e
        from_layer = self.from_layer.pop()
        if from_layer is None: return
        r = np.einsum('bo,oi->bi', e, self.w, optimize=True)
        from_layer.backward(r)

    def step(self) -> None:
        self.w += np.einsum('bo,bi->oi', self.e, self.h, optimize=True) * self.lr
        self.b += np.einsum('bo->o', self.e, optimize=True) * self.lr

    def __call__(self, x: tuple[np.ndarray, Layer]) -> tuple[np.ndarray, Layer]:
        return self.forward(x)


class Conv2d(LearnLayer):
    def __init__(
            self,
            lr: float,
            in_channels: int,
            out_channels: int,
            kernel_size: int | tuple[int, int],
            stride: int | tuple[int, int] = (1, 1),
            padding: int = 0,
    ) -> None:
        super().__init__(lr)
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.kernel_size: tuple[int, int] = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 2
        self.stride: tuple[int, int] = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding: int = padding
        self.w: np.ndarray = np.random.rand(out_channels, in_channels, *self.kernel_size)
        self.history_input = None
        self.loss = None

    def forward(self, x: tuple[np.ndarray, Layer]) -> tuple[np.ndarray, Layer]:
        x = super().navigate(x)
        self.history_input = x
        x = np.pad(x, ((0, 0), (0, 0), *((self.padding,) * 2,) * 2))
        s = x.shape
        r = np.empty((s[0], self.out_channels,
                      (s[-2] - self.kernel_size[0] + 2 * self.padding) // self.stride[0] + 1,
                      (s[-1] - self.kernel_size[1] + 2 * self.padding) // self.stride[1] + 1))
        for b in range(s[0]):
            r[b] = signal.fftconvolve(x[b, None], self.w, 'valid')[:, 0]
        return r, self

    def backward(self, e: np.ndarray) -> None:
        self.loss = e
        from_layer = self.from_layer.pop()
        if from_layer is None:
            return
        r = np.zeros(self.history_input.shape)
        for b in range(e.shape[0]):
            for c in range(e.shape[1]):
                r[b] += signal.fftconvolve(e[b, c, None], self.w[c, :, ::-1])
        from_layer.backward(
            r if not self.padding else r[:, :, self.padding: -self.padding, self.padding: -self.padding])

    def step(self) -> None:
        for b in range(self.loss.shape[0]):
            self.w += signal.fftconvolve(self.loss[b, :, None], self.history_input[b, None], 'valid') * self.lr

    def __call__(self, x: tuple[np.ndarray, Layer]) -> tuple[np.ndarray, Layer]:
        return self.forward(x)


class MaxPool2d(Layer):
    def __init__(self, kernel_size: int or tuple[int, int],
                 stride: int or tuple[int, int] = (1, 1),
                 padding=0):
        super().__init__()
        self.in_shape: Cache = Cache()
        self.p: Cache = Cache()
        self.stride: tuple = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding: int = padding
        self.kernel_size: tuple = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)

    @staticmethod
    def max(x):
        return np.max(x), np.argmax(x)

    def forward(self, x: tuple[np.ndarray, Layer]) -> tuple[np.ndarray, Layer]:
        x = super().navigate(x)
        self.in_shape.push((x.shape[-2], x.shape[-1]))
        r = np.zeros((x.shape[0], x.shape[1],
                      (x.shape[-2] - self.kernel_size[0] + self.stride[1]) // self.stride[1],
                      (x.shape[-1] - self.kernel_size[1] + self.stride[0]) // self.stride[0]))
        p = r.copy()
        for b in np.arange(r.shape[0]):
            for t in np.arange(r.shape[1]):
                for i in np.arange(0, r.shape[2]):
                    for j in np.arange(0, r.shape[3]):
                        r[b, t, i, j], p[b, t, i, j] = self.max(
                            x[b, t, i * self.stride[1]: i * self.stride[1] + self.kernel_size[0],
                            j * self.stride[0]: j * self.stride[0] + self.kernel_size[1]])
        self.p.push(p)
        return r, self

    def backward(self, e: np.ndarray) -> None:
        from_layer = self.from_layer.pop()
        if from_layer is None:
            return
        r = np.zeros((e.shape[0], e.shape[1],
                      e.shape[-2] * self.stride[1] + self.kernel_size[0] - self.stride[1],
                      e.shape[-1] * self.stride[0] + self.kernel_size[1] - self.stride[0]))
        ps = self.p.pop()
        in_shape = self.in_shape.pop()
        for b in np.arange(0, e.shape[0]):
            for t in np.arange(0, e.shape[1]):
                for i in np.arange(0, e.shape[2]):
                    for j in np.arange(0, e.shape[3]):
                        p = (int(ps[b, t, i, j] // self.kernel_size[1]),
                             int(ps[b, t, i, j] % self.kernel_size[1]))
                        r[b, t, i * self.stride[1] + p[0], j * self.stride[0] + p[1]] = e[b, t, i, j]
                r[b, t] = np.pad(r[b, t], ((0, in_shape[-2] - r[b, t].shape[0]),
                                           (0, in_shape[-1] - r[b, t].shape[1])))
        from_layer.backward(r)

    def __call__(self, x: tuple[np.ndarray, Layer]) -> tuple[np.ndarray, Layer]:
        return self.forward(x)


class Flatten(Layer):
    def __init__(self, start_dim: int = 1, end_dim: int = -1):
        super().__init__()
        self.s = Cache()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: tuple[np.ndarray, Layer]) -> tuple[np.ndarray, Layer]:
        x = super().navigate(x)
        s = x.shape
        self.s.push(s)
        x = np.reshape(x, (*s[:self.start_dim], -1, *s[self.end_dim:][1:]))
        return x, self

    def backward(self, e: np.ndarray) -> None:
        from_layer = self.from_layer.pop()
        if from_layer is None:
            return
        from_layer.backward(e.reshape(self.s.pop()))

    def __call__(self, x: tuple[np.ndarray, Layer]) -> tuple[np.ndarray, Layer]:
        return self.forward(x)


class Relu(Layer):
    def __init__(self):
        super().__init__()
        self.h = Cache()

    def forward(self, x: tuple[np.ndarray, Layer]) -> tuple[np.ndarray, Layer]:
        x = super().navigate(x)
        self.h.push(x)
        x[x < 0] = 0
        return x, self

    def backward(self, e: np.ndarray) -> None:
        from_layer = self.from_layer.pop()
        if from_layer is None: return
        h = self.h.pop()
        e[h < 0] = 0
        from_layer.backward(e)

    def __call__(self, x: tuple[np.ndarray, Layer]) -> tuple[np.ndarray, Layer]:
        return self.forward(x)


class Sigmod(Layer):
    def __init__(self):
        super().__init__()
        self.o = Cache()

    def forward(self, x: tuple[np.ndarray, Layer]) -> tuple[np.ndarray, Layer]:
        x = super().navigate(x)
        self.o.push(scipy.special.expit(x))
        return self.o.peek(), self

    def backward(self, e: np.ndarray) -> None:
        from_layer = self.from_layer.pop()
        if from_layer is None: return
        o = self.o.pop()
        from_layer.backward(e * o * (1 - o))

    def __call__(self, x: tuple[np.ndarray, Layer]) -> tuple[np.ndarray, Layer]:
        return self.forward(x)


class Tanh(Layer):
    def __init__(self):
        super().__init__()
        self.o = Cache()

    def forward(self, x: tuple[np.ndarray, Layer]) -> tuple[np.ndarray, Layer]:
        x = super().navigate(x)
        self.o.push(np.tanh(x))
        return self.o.peek(), self

    def backward(self, e: np.ndarray) -> None:
        from_layer = self.from_layer.pop()
        if from_layer is None: return
        o = self.o.pop()
        from_layer.backward(e * (1 - o ** 2))

    def __call__(self, x: tuple[np.ndarray, Layer]) -> tuple[np.ndarray, Layer]:
        return self.forward(x)


class Start(Layer):
    def forward(self, x: object) -> None:
        pass

    def backward(self, e: object) -> None:
        pass


class Cache:
    def __init__(self):
        self.__items = []

    def __bool__(self):
        return self.__items != []

    def __iter__(self):
        return self.__items[::-1]

    def push(self, item):
        self.__items.append(item)

    def pop(self):
        if not self:
            return None
        return self.__items.pop()

    def peek(self):
        if not self:
            return None
        return self.__items[-1]

    def clear(self):
        self.__items.clear()
