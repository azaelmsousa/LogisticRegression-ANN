import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.datasets as sk_datasets
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, normalize

def softmax(y): 
    exp_y = np.exp(y)       
    return normalize(exp_y, norm='l1', axis=1)

#
# Cross entropy loss function, where h
# is the activation of the last layer.
# It computes the error of the predicted
# class and the correct one.
#
def cross_entropy(A, Y):
    A = softmax(A)
    eps = np.finfo(np.float64).eps
    A[A < eps]  = eps
    A[A > (1-eps)]  = 1-eps
    return ((-np.log(A) * Y) - (np.log(1-A) * (1-Y))).sum()
    #( (-np.log(A+eps)*Y) - (np.log(1-A+eps) * (1-Y)) ).sum()  
#
# Derivative of the Cross Entropy loss function.
# It is used in the back propagation
# algorithm.
#
def cross_entropy_derivative(I, A, Y):
    A = softmax(A)
    error = (A - Y)
    # print(Y.shape, A.shape, error.shape)
    grad = np.dot(I.T, error)    
    return grad

def cross_entropy_derivative_chain(A, Y):
    O = softmax(A)
    error = (O - Y)    
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
