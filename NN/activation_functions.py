import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.datasets as sk_datasets
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, normalize


# Sigmoid function for the activation
# of a neuron, where h is the dot
# product of X (input) and theta (weights)
#
def sigmoid(h):
    sig = 1. / (1. + np.exp(-h))
    return sig

#
# Derivative of the sigmoid function. It
# is used as part of the backpropagation
# algorithm
#
def sigmoid_derivative(h):
    sig = sigmoid(h)
    derivative = sig*(1-sig)
    return derivative

def sigmoid_derivative_chain(output):
    derivative = output*(1-output)
    return derivative

#
# Hyberbolic tangent function for the
# acivation of a neuron, where h is the
# dot product of X (input) and theta (weights)
#
def tanh(h):
    return (2 / (1+np.exp(-2*h)))-1

#
# Derivative of the hyperbolic tangent
# function. It is used as part of the
# back propagation algorithm.
#
def tanh_derivative(h):
    tanh_l = (4*np.exp(-2*h))/((1+np.exp(-2*h))**2)
    return tanh_l

def tanh_derivative_chain(h):    
    return (4*np.exp(-2*h))/((1+np.exp(-2*h))**2)


def relu(h):        
    return np.maximum(h, 0)

def relu_derivative_chain(h):
    v = h.copy()
    v[v > 0] = 1
    v[v <= 0] = 0.
    return  v

#
# The softmax function transforms a set of weights in to a probability distribution function
#
def softmax(y): 
    exp_y = np.exp(y - np.max(y))       
    return exp_y / exp_y.sum()

