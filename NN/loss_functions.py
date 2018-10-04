import math
import os
import sys
sys.path.append('../')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.datasets as sk_datasets
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
import NN.activation_functions as af
from sklearn.preprocessing import LabelBinarizer, normalize

def softmax(y): 
     return af.softmax(y)

# Cross entropy loss function, where h
# is the activation of the last layer.
# It computes the error of the predicted
# class and the correct one.
#
def cross_entropy(p, y):
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)
    return - y * np.log(p) - (1 - y) * np.log(1 - p)


def cross_entropy_derivative_chain(p, y):            
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)
    return - (y / p) + (1 - y) / (1 - p)
#
# Sum of the squared differences (SMD) loss
# function, where h is the activation of the
# last layer. It computes the error of the
# predicted class and the correct one.
#
def smd(p, y):
    error = np.square((p - y))
    return error/2

#
# Sum of the squared differences (SMD) loss
# function, where h is the activation of the
# last layer. It computes the error of the
# predicted class and the correct one.
#

def smd_derivative_chain(p, y):
    return -(y - p)

