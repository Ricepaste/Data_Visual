#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import sys
import os

import math
import matplotlib.pyplot as plt
import numpy
import pandas
import sklearn.metrics
import sklearn.model_selection
import sklearn.linear_model
import sklearn.preprocessing


def load_train_test_data(train_ratio=.5):
    # var共有8維，最後一個是label
    data = pandas.read_csv('./HTRU_2.csv', header=None,
                           names=['x%i' % (i) for i in range(8)] + ['y'])
    X = numpy.asarray(data[['x%i' % (i) for i in range(8)]])

    # 增加bias項，X從8維變成9維(n*8 -> n*9)
    # hstack: Stack arrays in sequence horizontally (column wise).
    X = numpy.hstack((numpy.ones((X.shape[0], 1)), X))

    # y是1維(n*1)
    y = numpy.asarray(data['y'])

    return sklearn.model_selection.train_test_split(X, y, test_size=1 - train_ratio, random_state=0)


def scale_features(X_train, X_test, low=0, upp=1):
    # 歸一化到[low, upp]區間
    # fit(numpy.vstack((X_train, X_test))) -> 找出X_train和X_test的最大值和最小值
    minmax_scaler = sklearn.preprocessing.MinMaxScaler(
        feature_range=(low, upp)).fit(numpy.vstack((X_train, X_test)))

    X_train_scale = minmax_scaler.transform(X_train)
    X_test_scale = minmax_scaler.transform(X_test)

    return X_train_scale, X_test_scale


def cross_entropy(y, y_hat):
    loss = 0
    for i in range(y.size):
        loss += -(y[i]*math.log(y_hat[i]) + (1-y[i])*math.log(1-y_hat[i]))
    return loss/y.size


def logreg_sgd(X, y, alpha=.001, epochs=10000, eps=1e-4):
    # alpha: step size
    # epochs: max epochs
    # eps: stop when the thetas between two epochs are all less than eps
    n, d = X.shape
    y = y.reshape((n, 1))

    theta = numpy.zeros((d, 1))
    theta_old = copy.deepcopy(theta)

    epoch_list = []
    loss_list = []

    # 正向傳播
    for epoch in range(epochs):
        for batch in range(n):
            y_hat = predict_prob(X[batch], theta)
            # 反向傳播
            theta = theta - alpha * numpy.dot((X[batch].T).reshape(9, 1),
                                              (y_hat - y[batch]).reshape(1, 1))

        if (epoch % 10 == 0):
            y_hat = predict_prob(X, theta)
            epoch_list.append(epoch)
            loss_list.append(cross_entropy(y, y_hat).item())
            print("Epoch {}:\tLoss: {}".format(
                epoch, cross_entropy(y, y_hat).item()))

        if numpy.all(numpy.abs(theta - theta_old) < eps):
            break

        theta_old = copy.deepcopy(theta)

    plt.plot(epoch_list, loss_list)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xlim(0, epochs)
    plt.title("alpha = {}, eps = {}".format(alpha, eps))

    # files = os.listdir()
    # name = "loss.png"
    # index = 0
    # while name in files:
    #     index += 1
    #     name = "loss_{}.png".format(index)

    # plt.savefig(name)
    plt.show()
    # plt.close()

    return theta


def predict_prob(X, theta):
    # n*9 dot 9*1 -> n*1
    # 一層神經網路+sigmoid
    return 1./(1+numpy.exp(-numpy.dot(X, theta)))


def plot_roc_curve(y_test, y_prob, alpha, eps):
    # TODO: compute tpr and fpr of different thresholds
    tpr = []
    fpr = []

    for threshold in numpy.arange(0, 1, 0.01):
        y_pred = y_prob > threshold
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for i in range(len(y_test)):
            if y_pred[i] == True and y_test[i] == True:
                tp += 1
            elif y_pred[i] == True and y_test[i] == False:
                fp += 1
            elif y_pred[i] == False and y_test[i] == True:
                fn += 1
            elif y_pred[i] == False and y_test[i] == False:
                tn += 1
        tpr.append(tp/(tp+fn))
        fpr.append(fp/(fp+tn))

    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("alpha = {}, eps = {}".format(alpha, eps))
    plt.gca().set_aspect('equal', adjustable='box')

    # files = os.listdir()
    # name = "roc_curve.png"
    # index = 0
    # while name in files:
    #     index += 1
    #     name = "roc_curve_{}.png".format(index)

    # plt.savefig(name)
    plt.show()
    # plt.close()


def main(argv):
    # 讀取資料，並切分成train set和test set各半，接著將特徵做歸一化到[0, 1]區間
    X_train, X_test, y_train, y_test = load_train_test_data(train_ratio=.5)
    X_train_scale, X_test_scale = scale_features(X_train, X_test, 0, 1)

    alpha_now = 0.001
    eps_now = 1e-1

    theta = logreg_sgd(X_train_scale, y_train,
                       alpha=alpha_now, eps=eps_now, epochs=10000)
    print(theta)

    y_train = y_train.reshape((y_train.shape[0], 1))
    y_prob = predict_prob(X_train_scale, theta)

    print("Logreg train accuracy: %f" %
          (sklearn.metrics.accuracy_score(y_train, y_prob > .5)))
    print("Logreg train precision: %f" %
          (sklearn.metrics.precision_score(y_train, y_prob > .5)))
    print("Logreg train recall: %f" %
          (sklearn.metrics.recall_score(y_train, y_prob > .5)))
    y_prob = predict_prob(X_test_scale, theta)
    print("Logreg test accuracy: %f" %
          (sklearn.metrics.accuracy_score(y_test, y_prob > .5)))
    print("Logreg test precision: %f" %
          (sklearn.metrics.precision_score(y_test, y_prob > .5)))
    print("Logreg test recall: %f" %
          (sklearn.metrics.recall_score(y_test, y_prob > .5)))

    plot_roc_curve(y_test.flatten(), y_prob.flatten(),
                   alpha_now, eps_now)


if __name__ == "__main__":
    main(sys.argv)
