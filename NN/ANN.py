import sys
sys.path.append('../')

import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.datasets as sk_datasets
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from NN import activation_functions

class ANN:

    def __init__(self, activation):
        self.act_func = activation_functions.__dict__[activation]
        self.act_derivative = activation_functions.__dict__[
            activation+"_derivative"]


    def initialize_random_weights(self, n_input, n_perceptron, n_classes):
        self.n_hidden_layers = len(n_perceptron)
        self.hidden_layers = []

        for l in range(self.n_hidden_layers):
            if (l == 0):
                w = np.random.rand(n_input+1, n_perceptron[0])
            else:
                w = np.random.rand(n_perceptron[l-1]+1, n_perceptron[l])
            self.hidden_layers.append(w)

        if (self.n_hidden_layers == 0):
            self.output_layer = np.random.rand(n_input+1, n_classes)
        else:
            self.output_layer = np.random.rand(
                n_perceptron[self.n_hidden_layers-1]+1, n_classes)

    def initialize_fixed_weights(self, w):
        self.hidden_layers = w[:-1]
        self.output_layer = w[-1]
        self.n_hidden_layers = len(w)-1

    def show_weights(self):

        for l in range(self.n_hidden_layers):
            print("Hidden Layer ", str(l+1))
            print(self.hidden_layers[l], "\n")

        print("Output Layer ")
        print(self.output_layer, "\n")

    def show_setup(self):

        print(" Input size: ", str(self.hidden_layers[0].shape[0]-1))
        print(" Number of hidden layers: ", str(self.n_hidden_layers))
        print(" Number of perceptrons at each layer: ")
        for l in range(self.n_hidden_layers):
            print(" HL "+str(l+1)+": " +
                  str(self.hidden_layers[l].shape[1]))
        print(" Number of classes: "+str(self.output_layer.shape[1]), "\n")

    def foward_propagation(self, X):

        inp = np.insert(X, 0, 1, axis=1)

        for l in range(self.n_hidden_layers):
            out = np.matmul(inp, self.hidden_layers[l])
            sig = self.act_func(out)
            inp = np.insert(sig, 0, 1, axis=1)

        out = np.matmul(inp, self.output_layer)
        sig = self.act_func(out)

        return out, sig
