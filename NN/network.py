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

DEBUG = False


def dprint (value):
    if DEBUG:
        print(value)


class Layer:
    def __init__(self, input_sz, neurons, activation, bias=1.0, weights=None, label=None):
        self.activation = activation_functions.__dict__[activation]
        self.input_sz = input_sz
        self.neurons = neurons
        self.label = label
        self.bias = bias
        self.model = None
        if activation != 'softmax':
            self.act_derivative = activation_functions.__dict__[
                activation+"_derivative_chain"]
            if weights is not None:
                self.weights = weights
            else:
                self.weights = np.random.rand(input_sz, neurons)
        else:
            self.weights = np.ones((input_sz, neurons))

    def set_model(self, model):
        self.model = model

    def __str__(self):
        return inspect.cleandoc("""
        {}\t(input={}, neurons={}, activation={})""".format(self.label, self.input_sz, self.neurons, self.activation.__name__))

    def feed_forward(self, input_values):

        if self.activation.__name__ != 'softmax':
            _input_values = input_values.copy()

            W = self.weights.copy()
            B = np.ones(input_values.shape[0])*self.bias
            dprint((self.label, _input_values.shape, W.shape))
            self.net = np.matmul(_input_values, W) + B
        else:
            self.net = input_values.copy()

        self.input = input_values
        self.out = self.activation(self.net)

        dprint(('net', self.net))
        dprint(('out', self.out))
        return self.net, self.out

    # Here starts the back propagation implementation
    def update(self):
        if self.model is None: 
            lr = 0.5
        else: 
            lr  = self.model.lr
        self.weights -= (lr * self.dW)

        dprint(('self.weights',self.weights))

    def set_dOut_dNet(self):
        self.dOut_dNet = self.act_derivative(self.out)
        dprint(('self.dOut_dNet',self.dOut_dNet))

    def set_delta(self, dETotal_dOut):
        self.delta = dETotal_dOut * self.dOut_dNet
        dprint(('self.delta',self.delta))

    def set_dW(self):
        self.dW = self.delta * self.input
        dprint(('self.dW',self.dW))

    def set_dETotal_dOut(self):
        temp  = (self.delta * self.weights)
        dprint(('calc_dETotal_dOut', temp))
        self.dETotal_dOut = temp.sum(axis=1)

    def get_dETotal_dOut(self):
        dprint(('get_dETotal_dOut', self.dETotal_dOut))
        return self.dETotal_dOut

    def backpropagate(self, output_layer=None, dETotal_dOut=None):
        if output_layer is None:
            dETotal_dOut = dETotal_dOut
        else:
            dETotal_dOut = output_layer.get_dETotal_dOut()

        dprint(('dETotal_dOut',dETotal_dOut))

        self.set_dOut_dNet()
        self.set_delta(dETotal_dOut)
        self.set_dW()
        # update the total derivative error before weight updates
        self.set_dETotal_dOut()

        self.update()


class NN:
    def __init__(self, loss, lr=0.01):
        self.clear_layers()
        self.lr = lr
        self.loss = loss_functions.__dict__[loss]
        self.loss_derivative = loss_functions.__dict__[loss+"_derivative"]

    def clear_layers(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def summary(self):
        dprint(("Model Summary"))
        dprint(("-------------------------------"))
        for layer in self.layers:
            dprint((layer))
        dprint(("-------------------------------"))

    def show_weights(self):
        dprint(("Model Weights"))
        dprint(("-------------------------------"))
        for layer in self.layers:
            dprint((layer))
            dprint((layer.weights))
        dprint(("-------------------------------"))

    def feed_forward(self, X):
        input_values = X
        for layer in self.layers:
            net, out = layer.feed_forward(input_values)
            input_values = out

        return net, out

    def fit(self, X, y, max_it=100):
        it = 0
        while it < max_it:
            _, pdf = self.feed_forward(X)
            y_ = pdf.argmax(axis=-1)

            it += 1
