# _*_ coding: utf-8 _*_

"""
An implementation of multi-layer perceptron using only Nummy by Denny Britiz.

Author: Genpeng Xu
Date:   2019/03/16
"""

import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons, make_classification, make_circles
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression


if __name__ == '__main__':
    m = 5000  # the number of samples
    rng = np.random.RandomState(seed=89)
    X, y = make_moons(m, noise=0.3, random_state=rng)
    # X, y = make_classification(m, n_features=2, n_redundant=0,
    #                            n_informative=2, random_state=rng, n_clusters_per_class=1)
    # X, y = make_circles(m, noise=.2, factor=.5, random_state=rng)

    # =============================================================================================== #
    # Split the dataset into training set and test set

    df = pd.DataFrame(X, columns=['x_1', 'x_2'])
    df['label'] = y

    df_pos = df.loc[df.label == 0]
    df_neg = df.loc[df.label == 1]

    n_pos_train, n_neg_train = int(len(df_pos) * 0.7), int(len(df_neg) * 0.7)
    df_pos_train = df_pos[:n_pos_train]
    df_pos_test = df_pos[n_pos_train:]
    df_neg_train = df_neg[:n_neg_train]
    df_neg_test = df_neg[n_neg_train:]

    df_tmp = pd.concat([df_pos_train, df_neg_train], axis=0)
    df_test = pd.concat([df_pos_test, df_neg_test], axis=0)

    del df_pos, df_neg, df_pos_train, df_pos_test, df_neg_train, df_neg_test
    gc.collect()

    # =============================================================================================== #

    # =============================================================================================== #
    # Split the training set into training and validation set

    df_tmp_pos = df_tmp.loc[df_tmp.label == 0]
    df_tmp_neg = df_tmp.loc[df_tmp.label == 1]

    n_pos_train, n_neg_train = int(len(df_tmp_pos) * 0.7), int(len(df_tmp_neg) * 0.7)
    df_tmp_pos_train = df_tmp_pos[:n_pos_train]
    df_tmp_pos_val = df_tmp_pos[n_pos_train:]
    df_tmp_neg_train = df_tmp_neg[:n_neg_train]
    df_tmp_neg_val = df_tmp_neg[n_neg_train:]

    df_train = pd.concat([df_tmp_pos_train, df_tmp_neg_train], axis=0)
    df_val = pd.concat([df_tmp_pos_val, df_tmp_neg_val], axis=0)

    del df_tmp, df_tmp_pos, df_tmp_neg, df_tmp_pos_train, df_tmp_pos_val, df_tmp_neg_train, df_tmp_neg_val
    gc.collect()

    # =============================================================================================== #

    # =============================================================================================== #
    # Select X and y

    X_train, y_train = df_train.iloc[:, :2].values, df_train.iloc[:, 2].values
    X_val, y_val = df_val.iloc[:, :2].values, df_val.iloc[:, 2].values
    X_test, y_test = df_test.iloc[:, :2].values, df_test.iloc[:, 2].values

    del df_train, df_val, df_test
    gc.collect()

    # =============================================================================================== #

    h = 0.02
    x_min, x_max = X_train[:, 0].min() - .5, X_train[:, 0].max() + .5
    y_min, y_max = X_train[:, 1].min() - .5, X_train[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    names = ['K-Nearest Neighbors',
             'Naive Bayes',
             'Logistic Regression',
             'Support Vector Machine',
             'Decision Tree',
             'Multi-Layer Perceptron']
    classifiers = [KNeighborsClassifier(3),
                   GaussianNB(),
                   LogisticRegression(),
                   SVC(gamma=2),
                   DecisionTreeClassifier(max_depth=5),
                   MLPClassifier(hidden_layer_sizes=5,
                                 activation='tanh',
                                 solver='sgd',
                                 max_iter=10000)]

    figure = plt.figure(figsize=(16, 32))
    i = 1
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(3, 2, i)
        ax.set_title(name)
        clf.fit(X_train, y_train)
        score_val = clf.score(X_val, y_val)
        score_test = clf.score(X_test, y_test)

        # plot the decision boundary
        if hasattr(clf, 'decision_function'):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], s=40, c=y_train, cmap=cm_bright, edgecolors='k')
        # plot the validation points
        ax.scatter(X_val[:, 0], X_val[:, 1], s=40, c=y_val, cmap=cm_bright, edgecolors='k', alpha=.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        ax.text(xx.max() - .3, yy.min() + .5, ('%.2f' % score_val).lstrip('0'), size=15, horizontalalignment='right')
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score_test).lstrip('0'), size=15, horizontalalignment='right')


        i += 1

    plt.show()
