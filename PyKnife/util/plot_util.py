# _*_ coding: utf-8 _*_

"""
Some utility function used for plotting.

Author: Genpeng Xu
Date:   2019/03/14
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_decision_boundary(X, y, pred_func, p=0.5, h=0.01):
    x_min, x_max = X[:, 0].min() - p, X[:, 0].max() + p
    y_min, y_max = X[:, 1].min() - p, X[:, 1].max() + p
    # generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # predict the function value for the whole grid
    z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    plt.contourf(xx, yy, z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
