import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.datasets as sk_datasets
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, normalize
from math import exp, log


def softmax(x):
    assert len(x.shape) == 2
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def softmax_derivative_chain(x): 
    p = softmax(x)
    return p*(1-p)

# Sigmoid function for the activation
# of a neuron, where h is the dot
# product of X (input) and theta (weights)
#
def sigmoid(h):    
    sig = 1. / (1. + np.exp(-h))    
    return sig


def sigmoid_derivative_chain(h):
    output = sigmoid(h)
    derivative = output*(1-output)
    return derivative

#
# Hyberbolic tangent function for the
# acivation of a neuron, where h is the
# dot product of X (input) and theta (weights)
#
def tanh(h):
    h[h<=0] = np.finfo(np.float128).eps
    return (2 / (1+np.exp(-2*h)))-1

#
# Derivative of the hyperbolic tangent
# function. It is used as part of the
# back propagation algorithm.
#
def tanh_derivative(h):
    h[h<=0] = np.finfo(np.float128).eps
    tanh_l = (4*np.exp(-2*h))/((1+np.exp(-2*h))**2)
    return tanh_l

def tanh_derivative_chain(h):    
    h[h<=0] = np.finfo(np.float128).eps
    return (4*np.exp(-2*h))/((1+np.exp(-2*h))**2)


def relu(x):        
    return np.where(x >= 0, x, 0)

def relu_derivative_chain(x):    
    return  np.where(x >= 0, 1, 0)

#
# The softmax function transforms a set of weights in to a probability distribution function
#

