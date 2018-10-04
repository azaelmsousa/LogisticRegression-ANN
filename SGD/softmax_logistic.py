import sys
sys.path.append("../")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import sklearn.metrics as metrics
import sklearn.datasets as sk_datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import normalize
from utils import dataset_helper

# -----------------------------------
#   Softmax Classify Methods
# -----------------------------------


def classify_softmax(theta, X):
    X = np.insert(X, 0, 1, axis=1)
    h = softmax(theta, X)
    pred = np.argmax(h, axis=0)
    X = np.delete(X, 0, axis=1)
    return pred

# -----------------------------------
#   Softmax Training Methods
# -----------------------------------


def softmax(theta, X):
    score = np.dot(theta, X.transpose())
    exp = np.exp(score)
    h = exp / np.sum(exp, axis=0)
    return h


def Cost(theta, X, y):
    nsamples = X.shape[0]
    h = softmax(theta, X)
    lnh = np.log(h)
    ylnh = np.sum((np.multiply(lnh, y)), axis=0)
    error = np.sum(ylnh)/-nsamples
    return error

# -----------------------------------
#   Evaluation Metrics
# -----------------------------------


def AccuracyScore(Y, predY, mode='binary'):
    acc = 0.0
    if (mode == 'binary'):
        TP = ((predY == Y) & (predY == 1.)).sum()
        TN = ((predY == Y) & (predY == 0.)).sum()
        acc = (TP + TN) / Y.shape[0]
    elif (mode == 'multi'):
        TP = (predY == Y).sum()
        acc = TP / Y.shape[0]
    return acc


def PrecisionScore(Y, predY, mode='binary'):
    precision = 0.0
    if (mode == 'binary'):
        TP = ((predY == Y) & (predY == 1)).sum()
        FP = ((predY != Y) & (predY == 1)).sum()
        precision = TP / (TP + FP)
    elif (mode == 'multi'):
        classes = np.unique(Y)
        for c in classes:
            TP = ((predY == Y) & (predY == c)).sum()
            FP = ((predY != Y) & (predY == c)).sum()
            precision += TP / (TP + FP)
        precision /= len(classes)
    return precision


def RecallScore(Y, predY, mode='binary'):
    recall = 0.0
    if (mode == 'binary'):
        TP = ((predY == Y) & (predY == 1)).sum()
        FN = ((predY != Y) & (predY == 0)).sum()
        recall = TP / (TP + FN)
    elif (mode == 'multi'):
        classes = np.unique(Y)
        for c in classes:
            TP = ((predY == Y) & (predY == c)).sum()
            FN = ((predY != Y) & (Y == c)).sum()
            recall += TP / (TP + FN)
        recall /= len(classes)
    return recall


def FbScore(Y, predY, beta, mode='binary'):
    fbscore = 0.0
    if (mode == 'binary'):
        precision = PrecisionScore(predY, Y)
        recall = RecallScore(predY, Y)
        fscore = (1 + beta*beta)*((precision*recall) /
                                  ((beta*beta*precision)+recall))
    elif (mode == 'multi'):
        precision = PrecisionScore(predY, Y, 'multi')
        recall = RecallScore(predY, Y, 'multi')
        fscore = (1 + beta*beta)*((precision*recall) /
                                  ((beta*beta*precision)+recall))
    return fscore

# -----------------------------------
#   Gradient Descent
# -----------------------------------


def BGD(X, y, alpha, iterations):

    X = np.insert(X, 0, 1, axis=1)

    lb = LabelBinarizer()
    lb.fit(y)
    y_enc = (lb.transform(y)).transpose()

    nsamples = X.shape[0]
    nfeatures = X.shape[1]

    print(nsamples)
    print(nfeatures)

    theta = np.zeros([len(np.unique(y)), nfeatures])
    J = []

    for i in range(iterations):

        h = softmax(theta, X)

        error = h - y_enc

        grad = (np.matmul(error, X))/nsamples

        theta = theta - (alpha*grad)

        J.append(Cost(theta, X, y_enc))

        print("Iteration:", i, "Cost=", Cost(theta, X, y_enc))

    X = np.delete(X, 0, axis=1)

    plt.plot(J)
    plt.ylabel('Error')
    plt.xlabel('iterations')
    plt.show()

    return theta, J[iterations-1]
