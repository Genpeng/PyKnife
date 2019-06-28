# _*_ coding: utf-8 _*_

"""
Implementing a multi-layer perceptron (MLP) from scrach.

Author: Genpeng Xu
Date:   2019/03/14
"""

# Import necessary packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV

# Import customized modules
from PyKnife.math.functions import softmax
from PyKnife.util.plot_util import plot_decision_boundary


def init_params_normal(n_x, n_h, n_y):
    """Initialize all the weights and biases with Gaussian distribution."""
    np.random.seed(2)
    W1 = np.random.randn(n_h, n_x) / np.sqrt(n_x)
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) / np.sqrt(n_h)
    b2 = np.zeros((n_y, 1))
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}


def init_params_uniform(n_x, n_h, n_y):
    """Initialize all the weights and biases with Uniform distribution."""
    np.random.seed(2)
    W1 = np.random.uniform(-1 / np.sqrt(n_x), 1 / np.sqrt(n_x), size=(n_h, n_x))
    b1 = np.zeros((n_h, 1))
    W2 = np.random.uniform(-1 / np.sqrt(n_h), 1 / np.sqrt(n_h), size=(n_y, n_h))
    b2 = np.zeros((n_y, 1))
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}


def forward_propagation(X, params):
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    # forward propagation
    Z1 = np.dot(W1, X.T) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)

    return A2, {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}


def backward_propagation(X, Y, cache, params, reg_lambda=0.01):
    A2 = cache['A2']
    A1 = cache['A1']

    W2 = params['W2']
    W1 = params['W1']

    # backpropagation
    dZ2 = A2 - Y.T
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dZ1 = np.dot(W2.T, dZ2) * (1 - A1)
    dW1 = np.dot(dZ1, X) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    # add regularization terms
    dW2 += reg_lambda * W2
    dW1 += reg_lambda * W1

    return {'dW2': dW2, 'db2': db2, 'dW1': dW1, 'db1': db1}


def update_params(params, grads, learning_rate=0.1):
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']

    # gradient descent parameter update
    W2 += -learning_rate * dW2
    b2 += -learning_rate * db2
    W1 += -learning_rate * dW1
    b1 += -learning_rate * db1

    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}


def compute_loss(Y, A2, params, reg_lambda=0.01):
    W1 = params['W1']
    W2 = params['W2']
    data_loss = np.sum(Y.T * np.log(A2)) / (-m)
    reg_term = reg_lambda * (np.sum(np.square(W2)) + np.sum(np.square(W1))) / 2
    return data_loss + reg_term


def predict(X, params):
    probs, _ = forward_propagation(X, params)
    return np.argmax(probs, axis=0).ravel()


def nn_model_scratch(X, Y, n_h=5, reg_lambda=0.01, learning_rate=0.1, epochs=10000, print_loss=False):
    np.random.seed(89)

    n_x = X.shape[1]  # the shape of X is (m, n)
    n_y = Y.shape[1]  # the shape of Y is (m, q)

    # Initialize parameters
    W1 = np.random.randn(n_h, n_x) / np.sqrt(n_x)
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) / np.sqrt(n_h)
    b2 = np.zeros((n_y, 1))

    for i in range(1, epochs + 1):
        # forward propagation
        Z1 = np.dot(W1, X.T) + b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = softmax(Z2)

        # backpropagation
        dZ2 = A2 - Y.T
        dW2 = np.dot(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m
        dZ1 = np.dot(W2.T, dZ2) * (1 - A1)
        dW1 = np.dot(dZ1, X) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        # add regularization terms
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1

        # gradient descent parameter update
        W2 += -learning_rate * dW2
        b2 += -learning_rate * db2
        W1 += -learning_rate * dW1
        b1 += -learning_rate * db1

        if print_loss and i % 1000 == 0:
            data_loss = np.sum(Y.T * np.log(A2)) / (-m)
            reg_term = reg_lambda * (np.sum(np.square(W2)) + np.sum(np.square(W1))) / 2
            loss = data_loss + reg_term
            print("Loss after iteration %6d: %.6f" % (i, loss))

    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}


def nn_model(X, Y, n_h=5, reg_lambda=0.01, learning_rate=0.1, epochs=10000, print_loss=False):
    n_x = X.shape[1]  # the shape of X is (m, n)
    n_y = Y.shape[1]  # the shape of Y is (m, q)

    # Initialize parameters
    params = init_params_normal(n_x, n_h, n_y)

    for i in range(1, epochs + 1):
        # forward propagation
        A2, cache = forward_propagation(X, params)

        # backpropagation
        grads = backward_propagation(X, Y, cache, params, reg_lambda=reg_lambda)

        # gradient descent parameter update
        params = update_params(params, grads, learning_rate=learning_rate)

        if print_loss and i % 100 == 0:
            loss = compute_loss(Y, A2, params)
            print("Loss after iteration %6d: %.6f" % (i, loss))

    return params


if __name__ == '__main__':
    # Generate a dataset
    m = 200  # number of samples
    rng = np.random.RandomState(89)
    X, y = make_moons(m, noise=0.2, random_state=rng)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=rng)

    # Change to categorical label
    m_train = len(y_train)
    Y_train = np.zeros((m_train, 2), dtype=np.int8)
    Y_train[range(m_train), y_train] = 1

    # ============================================================================================== #
    # Classify by using Logistic Regression

    # clf = LogisticRegressionCV()
    # clf.fit(X_train, y_train)
    #
    # plot_decision_boundary(X_train, y_train, lambda x: clf.predict(x))
    # plt.title("Logistic Regression")
    # plt.show()
    #
    # preds_test = clf.predict(X_test)
    # acc_lr = (np.sum(y_test * preds_test) + np.sum((1 - y_test) * (1 - preds_test))) / X_test.shape[0]
    # print("The accuracy of Logistic Regression is: %.4f" % acc_lr)

    # ============================================================================================== #

    # ============================================================================================== #
    # Classify by using Multi-Layer Perceptron

    # params = nn_model(X_train, Y_train, n_h=4, reg_lambda=0.01,
    #                   learning_rate=0.01, epochs=20000, print_loss=True)
    params = nn_model_scratch(X_train, Y_train, n_h=5, reg_lambda=0.01,
                              learning_rate=0.01, epochs=20000, print_loss=True)

    plot_decision_boundary(X_train, y_train, lambda x: predict(x, params))
    plt.title("Multi-Layer Perceptron")
    plt.show()

    preds_test = predict(X_test, params)
    acc_mlp = (np.sum(y_test * preds_test) + np.sum((1 - y_test) * (1 - preds_test))) / X_test.shape[0]
    print("The accuracy of Multi-Layer Perceptron is: %.4f" % acc_mlp)

    # ============================================================================================== #
