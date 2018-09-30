import sys
sys.path.append('../')

import math
import os

import inspect

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.datasets as sk_datasets
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from NN import activation_functions, loss_functions


class Layer:
    def __init__(self, input_sz, neurons, activation, bias=1.0, weights=None, label=None):
        self.activation = activation_functions.__dict__[activation]
        self.input_sz = input_sz
        self.neurons = neurons
        self.label = label
        self.bias = bias
        if activation != 'softmax':
            self.act_derivative = activation_functions.__dict__[
                activation+"_derivative"]
            if weights is not None:
                self.weights = weights
            else:
                self.weights = np.random.rand(input_sz, neurons)
        else:
            self.weights = np.ones((input_sz, neurons))

    def __str__(self):
        return inspect.cleandoc("""
        {}\t(input={}, neurons={}, activation={})""".format(self.label, self.input_sz, self.neurons, self.activation.__name__))

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights

    def feed_forward(self, input_values):

        if self.activation.__name__ != 'softmax':
            _input_values = input_values.copy()

            W = self.get_weights().copy()
            B = np.ones(input_values.shape[0])*self.bias
            print(self.label, _input_values.shape, W.shape)
            net = np.matmul(_input_values, W) + B
        else:
            net = input_values.copy()

        out = self.activation(net)
        print('net', net)
        print('out', out)
        return net, out


class NN:
    def __init__(self, loss):
        self.clear_layers()
        self.loss = loss_functions.__dict__[loss]
        self.loss_derivative = loss_functions.__dict__[loss+"_derivative"]

    def clear_layers(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def summary(self):
        print("Model Summary")
        print("-------------------------------")
        for layer in self.layers:
            print(layer)
        print("-------------------------------")

    def show_weights(self):
        print("Model Weights")
        print("-------------------------------")
        for layer in self.layers:
            print(layer)
            print(layer.get_weights())
        print("-------------------------------")

    def feed_forward(self, X):
        input_values = X
        for layer in self.layers:
            net, out = layer.feed_forward(input_values)
            input_values = out

        return net, out

    def back_propagante(error):
        return None

    def fit(self, X, y, max_it=100):
        it = 0
        while it < max_it:
            _, pdf = self.feed_forward(X)
            y_ = pdf.argmax(axis=-1)

            it += 1
