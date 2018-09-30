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
def cross_entropy(A, Y):
    eps = np.finfo(np.float128).eps
    A[A < eps] = eps
    A[A > 1.-eps] = 1.-eps
    return (-np.multiply(np.log(A), Y) - np.multiply((np.log(1-A)), (1-Y))).sum()

#
# Derivative of the Cross Entropy loss function.
# It is used in the back propagation
# algorithm.
#
def cross_entropy_derivative(I, A, Y):
    error = (A - Y)
    grad = np.dot(I.T, error)
    return grad

def cross_entropy_derivative_chain(A, Y):
    error = (A - Y)
    return error
#
# Sum of the squared differences (SMD) loss
# function, where h is the activation of the
# last layer. It computes the error of the
# predicted class and the correct one.
#
def smd(A, Y):
    error = np.square((A - Y)).sum()
    return error/2

#
# Sum of the squared differences (SMD) loss
# function, where h is the activation of the
# last layer. It computes the error of the
# predicted class and the correct one.
#

def smd_derivative_chain(A, Y):
    return -(Y - A)

def smd_derivative(I, A, Y):
    error = (A - Y)
    grad = np.dot(I.transpose(), error)
    return grad

def mse(A, Y): 
    return ((A-Y)**2).mean()
#
# Mean Squared Error loss
# function, where h is the activation of the
# last layer. It computes the error of the
# predicted class and the correct one.
#
def mse_derivative(I, A, Y): 
    '''
        Return the gradient function for the MSE
        2/m * Sum(a - y)
    '''
    m = Y.shape[0]
    return (2/m) * (A - Y).sum()


def mse_derivative_chain(A, Y): 
    '''
        Return the gradient function for the MSE
        2/m * Sum(a - y)
    '''    
    return 2 * (A - Y)
