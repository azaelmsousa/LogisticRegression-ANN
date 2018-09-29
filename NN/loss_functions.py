import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.datasets as sk_datasets
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


#
# Cross entropy loss function, where h
# is the activation of the last layer.
# It computes the error of the predicted
# class and the correct one.
#
def cross_entropy(predY, y):
    eps = np.finfo(np.float128).eps
    predY[predY < eps] = eps
    predY[predY > 1.-eps] = 1.-eps
    return -np.multiply(np.log(predY), y) - np.multiply((np.log(1-predY)), (1-y))

#
# Derivative of the SMD loss function.
# It is used in the back propagation
# algorithm.
#
def cross_entropy_derivative(X, predY, y):
    error = (predY - y)
    grad = np.dot(X.transpose(), error)
    return grad

#
# Sum of the squared differences (SMD) loss
# function, where h is the activation of the
# last layer. It computes the error of the
# predicted class and the correct one.
#
def smd(predY, y):
    error = np.square((predY - y)).sum()
    return error/2

#
# Sum of the squared differences (SMD) loss
# function, where h is the activation of the
# last layer. It computes the error of the
# predicted class and the correct one.
#
def smd_derivative(X, predY, y):
    error = (predY - y)
    grad = np.dot(X.transpose(), error)
    return grad
