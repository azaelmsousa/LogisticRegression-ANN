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
from sklearn.utils import shuffle
from utils import dataset_helper

DEBUG = False

__iteration_log = []


def dprint(value):
    if DEBUG:
        print(value)


def insert_iteration_log(values):
    global __iteration_log
    __iteration_log.append(values)


def get_iteration_log():
    global __iteration_log
    cols = len(__iteration_log[0])
    print(cols)
    df = pd.DataFrame(__iteration_log, columns=[
                      'it', 'b_it', 'epoch', 'error_train', 'eta', 'error_val'][:cols])
    df.set_index(df.it)
    df.index = df.it
    return df


class Layer:
    def __init__(self, input_sz, neurons, activation, bias=.0, weights=None, label=None):
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
            self.net = np.dot(input_values, self.weights) + self.bias
        else:
            self.net = input_values.copy()

        self.input = input_values
        self.out = self.activation(self.net)

        dprint(('net', self.net))
        dprint(('out', self.out))
        return self.net, self.out

    def backpropagate(self, lr=0.5, output_layer=None, dETotal_dOut=None):

        if output_layer is None:
            self.error = dETotal_dOut
        else:
            self.error = output_layer.delta.dot(output_layer.weights.T)

        self.delta = self.error * self.act_derivative(self.out)

        self.weights -= lr * self.input.T.dot(self.delta)

        dprint(('dETotal_dOut', dETotal_dOut))


class NN:
    def __init__(self, loss, lr=0.01):
        self.clear_layers()
        self.lr = lr
        self.loss = loss_functions.__dict__[loss]
        self.loss_derivative = loss_functions.__dict__[
            loss+"_derivative_chain"]

    def clear_layers(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def summary(self):
        print(("Model Summary"))
        print(("-------------------------------"))
        for layer in self.layers:
            print((layer))
        print(("-------------------------------"))

    def show_weights(self):
        print(("Model Weights"))
        print(("-------------------------------"))
        for layer in self.layers:
            print((layer))
            print((layer.weights))
        print(("-------------------------------"))

    def feed_forward(self, X):
        input_values = X
        for layer in self.layers:
            net, out = layer.feed_forward(input_values)
            input_values = out

        return net, out

    def predict(self, X):
        result = [self.feed_forward(X[i:i+1, :])[1].flatten()
                  for i in range(X.shape[0])]
        return result

    def backpropagate(self, Y, lr=.5):
        l_sz = len(self.layers)
        l_idx = l_sz - 1
        layer = self.layers[l_idx]
        dE_dW = self.loss_derivative(layer.out, Y)
        layer.backpropagate(dETotal_dOut=dE_dW, lr=lr)

        while l_idx > 0:
            l_idx -= 1
            layer = self.layers[l_idx]
            dprint("============================")
            dprint(layer.label)
            dprint("============================")
            layer.backpropagate(output_layer=self.layers[l_idx+1])

    def fit(self, X, Y, lr=0.5,
            max_iter=1000, lr_optimizer=None,
            epsilon=0.001, power_t=0.25, t=1.0,
            print_interval=100, b_sz = 1,
            decay_iteractions=None, decay_rate=None,
            X_val=None,
            Y_val=None):

        error = 1
        it = 0
        epoch = 0
        lst_epoch = 0
        b_it = 0

        self.lr = lr

        

        X, Y = shuffle(X, Y)
        print('Shuffled')

        global __iteration_log
        __iteration_log = []
        error_val = 0.

        while ((error > epsilon) and (it < max_iter)):
            if (it % print_interval) == 0 or it == 1:
                error = .0
            if decay_iteractions is not None:
                eta = self.lr * math.pow(decay_rate, it // decay_iteractions)
            else:
                if lr_optimizer == 'invscaling':
                    eta = self.lr / (it + 1) * pow(t, power_t)
                else:
                    eta = self.lr

            X_ = np.zeros(0)
            Y_ = np.zeros(0)

            while Y_.shape[0] == 0:
                # Checking if it is a new epoch to shuffle the data.
                X_, Y_, b_it, epoch = dataset_helper.get_batch(
                    X, Y, b_it, b_sz, epoch)
                if lst_epoch < epoch:
                    lst_epoch = epoch
                    X, Y = shuffle(X, Y)

            # Only exits the loop when there is data for the batch

            # print(X_.shape)

            _, aY = self.feed_forward(X_)

            error += self.loss(aY, Y_) / Y_.shape[0]

            self.backpropagate(Y_, eta)

            it += 1
            t += 1

            if (it % print_interval) == 0:

                error /= print_interval

                if X_val is not None:
                    y_pred_val = np.array(self.predict(X_val))
                    error_val = np.array(
                        self.loss(y_pred_val, Y_val)) / X_val.shape[0]

                if X_val is not None:
                    print("It: %s Batch: %s Epoch %i Train Loss: %.8f lr: %.6f Val Loss: %.8f" %
                          (it, b_it, epoch, error, eta, error_val))
                else:
                    print("It: %s Batch: %s Epoch %i Error: %.8f lr: %.6f " %
                          (it, b_it, epoch, error, eta))

                if X_val is not None:
                    insert_iteration_log(
                        (it, b_it, epoch, error, eta, error_val))
                else:
                    insert_iteration_log((it, b_it, epoch, error, eta))
                last_error = error
                error = 1.0

        if X_val is not None:
            print("Finished \n It: %s Batch: %s Epoch %i Train Loss: %.8f lr: %.6f Val Loss: %.8f" %
                  (it, b_it, epoch, last_error, eta, error_val))
            insert_iteration_log(
                (it, b_it, epoch, last_error, eta, error_val))
        else:
            print("Finished \n It: %s Batch: %s Epoch %i Train Loss: %.8f lr: %.6f " %
                  (it, b_it, epoch, last_error, eta))
            insert_iteration_log((it, b_it, epoch, last_error, eta))
