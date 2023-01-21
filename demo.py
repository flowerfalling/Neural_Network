# -*- coding: utf-8 -*-
# @Time    : 2023/1/10 19:57
# @Author  : 之落花--falling_flowers
# @File    : demo.py
# @Software: PyCharm
import numpy as np
import torch


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features
    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        sample_mean = np.mean(x, axis=0, keepdims=True)
        sample_var = np.var(x, axis=0, keepdims=True)
        sample_sqrtvar = np.sqrt(sample_var + eps)
        x_norm = (x - sample_mean) / sample_sqrtvar
        out = x_norm * gamma + beta
        cache = (x, x_norm, gamma, beta, eps, sample_mean, sample_var, sample_sqrtvar)
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var

    elif mode == 'test':
        x_norm = (x - running_mean) / np.sqrt(running_var + eps)
        out = x_norm * gamma + beta

    # 将滑动均值和滑动方差保存或更新
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward_alt(dout, cache):
    dx, dgamma, dbeta = None, None, None

    N, D = dout.shape
    x, x_norm, gamma, beta, eps, sample_mean, sample_var, sample_sqrtvar = cache
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_norm, axis=0)
    dx_norm = dout * gamma
    dvar = np.sum((dx_norm * (x - sample_mean) * (-0.5) * np.power(sample_var + eps, -1.5)), axis=0)  # (D,)
    dmean = np.sum(dx_norm * (-1) * np.power(sample_var + eps, -0.5), axis=0)
    dmean += dvar * np.sum(-2 * (x - sample_mean), axis=0) / N
    dx = dx_norm * np.power(sample_var + eps, -0.5) + dvar * 2 * (x - sample_mean) / N + dmean / N

    return dx, dgamma, dbeta


def main():
    ...


if __name__ == '__main__':
    main()
