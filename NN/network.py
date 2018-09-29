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
from NN import activation_functions



class Layer:
    def __init__(self, input_sz, neurons, activation, weights=None, label=None):
        self.activation = activation_functions.__dict__[activation]
        
        self.input_sz = input_sz
        self.neurons = neurons
        self.label = label
        
        if activation != 'softmax':
            self.act_derivative = activation_functions.__dict__[
                activation+"_derivative"]
            if weights is not None:
                self.weights = weights
            else:
                self.weights = np.random.rand(input_sz+1, neurons)
        else:             
            self.weights = np.ones((input_sz+1, neurons))

    def __str__(self):
        return inspect.cleandoc("""
        {}\t(input={}, neurons={}, activation={})""".format(self.label,self.input_sz,self.neurons, self.activation.__name__))

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights


    def feed_forward(self, input_values):
        print("Forwardign {} weights {} {}".format(self.label, input_values.shape, self.weights.shape))

        if self.activation.__name__ != 'softmax':        
            _input_values = input_values.copy()
            _input_values = np.insert(_input_values, 0, 1, axis=1)
            output_values = np.matmul(_input_values, self.get_weights())
        else: 
            output_values = input_values

        activation_value = self.activation(output_values)
        
        return output_values, activation_value

class NN:    
    def __init__(self):
        self.clear_layers()

    def clear_layers(self):
        self.layers = []

    def add_layer(self, layer):        
        self.layers.append(layer)

    def summary(self):
        print("Model Summary")
        print("-------------------------------")
        for layer in self.layers: 
            print (layer)
        print("-------------------------------")

    def show_weights(self):
        print("Model Weights")
        print("-------------------------------")
        for layer in self.layers: 
            print (layer)            
            print (layer.get_weights())         
        print("-------------------------------")               
    
    def feed_forward(self, X):
        input_values = X        
        for layer in self.layers[:-1]:            
            output_values, activation_value = layer.feed_forward(input_values)            
            input_values = activation_value

        output_values, activation_value = self.layers[-1].feed_forward(input_values)            
        return output_values, activation_value




















