# _*_ coding: utf-8 _*_

"""
Implement an multi-layer perceptron (MLP) by referring to Denny Britz's code.

Author: Genpeng Xu
Date:   2019/03/17
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from PyKnife.math.functions import softmax_row


def init_params_normal(n_x, n_h, n_y):
    """Initialize all the weights and biases in the network."""
    np.random.seed(89)
    W1 = np.random.randn(n_x, n_h) / np.sqrt(n_x)
    b1 = np.zeros((1, n_h))
    W2 = np.random.randn(n_h, n_y) / np.sqrt(n_h)
    b2 = np.zeros((1, n_y))
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}


def init_params_uniform(n_x, n_h, n_y):
    """Initialize all the weights and biases in the network."""
    np.random.seed(89)
    W1 = np.random.uniform(-1 / np.sqrt(n_x), 1 / np.sqrt(n_x), size=(n_x, n_h))
    b1 = np.zeros(1, n_h)
    W2 = np.random.uniform(-1 / np.sqrt(n_h), 1 / np.sqrt(n_h), size=(n_h, n_y))
    b2 = np.zeros(1, n_y)
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}


def init_params_toy(n_x, n_h, n_y):
    """Initialize all the weights and biases in the network."""
    W1 = np.ones((n_x, n_h))
    b1 = np.zeros((1, n_h))
    W2 = np.ones((n_h, n_y))
    b2 = np.zeros((1, n_y))
    params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    return params


def forward_propagation_toy(X, params):
    W1, b1, W2, b2 = params['W1'], params['b1'], params['W2'], params['b2']
    Z1 = np.dot(X, W1) + b1
    A1 = Z1
    Z2 = np.dot(A1, W2) + b2
    A2 = Z2
    cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}
    return A2, cache


def forward_propagation(X, params):
    W1, b1, W2, b2 = params['W1'], params['b1'], params['W2'], params['b2']
    Z1 = np.dot(X, W1) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax_row(Z2)
    cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}
    return A2, cache


def backward_propagation(X, y, cache, params, reg_lambda=0.01):
    m = X.shape[0]  # number of samples
    A1, A2 = cache['A1'], cache['A2']
    W1, W2 = params['W1'], params['W2']
    # compute gradients of error with respect to parameters
    dZ2 = A2
    A2[range(m), y] -= 1
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    dZ1 = np.dot(dZ2, W2.T) * (1 - np.power(A1, 2))
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    # add regularization terms
    dW2 += reg_lambda * W2
    dW1 += reg_lambda * W1
    return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}


def update_params(params, grads, learning_rate=0.01):
    W1, b1, W2, b2 = params['W1'], params['b1'], params['W2'], params['b2']
    dW1, db1, dW2, db2 = grads['dW1'], grads['db1'], grads['dW2'], grads['db2']
    W1 += -learning_rate * dW1
    b1 += -learning_rate * db1
    W2 += -learning_rate * dW2
    b2 += -learning_rate * db2
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}


def compute_loss(X, y, params, reg_lambda=0.01):
    m = X.shape[0]  # number of samples
    W1, W2 = params['W1'], params['W2']
    probs, _ = forward_propagation(X, params)
    correct_logprobs = - np.log(probs[range(m), y])
    data_loss = np.sum(correct_logprobs) / m
    reg_term = reg_lambda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return data_loss + reg_term


def build_mlp(X, y, n_h=5, reg_lambda=0.01, learning_rate=0.01, epochs=20000, print_loss=True):
    n_x = X.shape[1]  # the number of nodes in the input layer
    n_y = np.max(y) + 1  # the number of nodes in the output layer

    # initialize all the weights and biases in the network
    params = init_params_normal(n_x, n_h, n_y)

    for i in range(1, 1 + epochs):
        # forward pass
        _, cache = forward_propagation(X, params)

        # backpropagation
        grads = backward_propagation(X, y, cache, params, reg_lambda=reg_lambda)

        # update parameters
        params = update_params(params, grads, learning_rate=learning_rate)

        if print_loss and i % 1000 == 0:
            loss = compute_loss(X, y, params, reg_lambda=reg_lambda)
            print("Loss after iteration %6d: %.4f" % (i, loss))

    return params


def plot_decision_boundary(X, y, pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    plt.contourf(xx, yy, Z, cmap=cm, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright, edgecolors='k')


def predict(X, params):
    probs, _ = forward_propagation(X, params)
    return np.argmax(probs, axis=1)


if __name__ == '__main__':
    # Generate a dataset
    num_samples = 20000  # number of samples
    rng = np.random.RandomState(89)
    X, y = make_moons(num_samples, noise=0.2, random_state=rng)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=rng)

    # Train and plot the decision boundary
    params = build_mlp(X_train, y_train, n_h=5, reg_lambda=0.001, learning_rate=0.1, epochs=20000)
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    plot_decision_boundary(X_train, y_train, lambda X: predict(X, params))
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors='k')
    plt.show()

    preds_train = predict(X_train, params)
    preds_test = predict(X_test, params)
    acc_train = (np.sum(preds_train * y_train) + np.sum((1 - preds_train) * (1 - y_train))) / X_train.shape[0]
    acc_test = (np.sum(preds_test * y_test) + np.sum((1 - preds_test) * (1 - y_test))) / X_test.shape[0]
    print()
    print("The accuracy of training set: %.4f" % acc_train)
    print("The accuracy of test set: %.4f" % acc_test)
