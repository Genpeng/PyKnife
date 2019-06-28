# _*_ coding: utf-8 _*_

"""
Some useful functions usually be used in machine learning.

Author: Genpeng Xu
Date:   2019/03/14
"""

import numpy as np


def sigmoid(Z):
    """Sigmoid function."""
    return 1 / (1 + np.exp(-Z))


def softmax_row(Z):
    """Softmax function."""
    exp_values = np.exp(Z)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def softmax_column(Z):
    """Softmax function."""
    exp_values = np.exp(Z)
    return exp_values / np.sum(exp_values, axis=0, keepdims=True)


if __name__ == '__main__':
    # 测试sigmoid函数
    z = 0
    print(sigmoid(z))

    # 测试softmax函数
    Z1 = np.array([[1, 1, 1],
                   [2, 2, 2],
                   [3, 3, 3]], dtype=np.int32)
    print(softmax_row(Z1))
