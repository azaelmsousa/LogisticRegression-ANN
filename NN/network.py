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


def dprint(value):
    if DEBUG:
        print(value)


__iteration_log = []


def get_iteration_log():
    global __iteration_log
    cols = len(__iteration_log[0])
    print(cols)
    df = pd.DataFrame(__iteration_log, columns=[
                      'it', 'b_it', 'epoch', 'error_train', 'eta', 'error_val'][:cols])
    df.set_index(df.it)
    return df


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
    def update(self, lr=0.5):
        self.weights -= (lr * self.dW)

        dprint(('self.weights', self.weights))

    def set_dOut_dNet(self):
        self.dOut_dNet = self.act_derivative(self.out)
        dprint(('self.dOut_dNet', self.dOut_dNet))

    def set_delta(self, dETotal_dOut):
        self.delta = dETotal_dOut * self.dOut_dNet
        dprint(('self.delta', self.delta))

    def set_dW(self):
        self.dW = self.delta * self.input
        dprint(('self.dW', self.dW))

    def set_dETotal_dOut(self):
        temp = (self.delta * self.weights)
        dprint(('calc_dETotal_dOut', temp))
        self.dETotal_dOut = temp.sum(axis=1)

    def get_dETotal_dOut(self):
        dprint(('get_dETotal_dOut', self.dETotal_dOut))
        return self.dETotal_dOut

    def backpropagate(self, lr=0.5, output_layer=None, dETotal_dOut=None):
        if output_layer is None:
            dETotal_dOut = dETotal_dOut
        else:
            dETotal_dOut = output_layer.get_dETotal_dOut()

        dprint(('dETotal_dOut', dETotal_dOut))

        self.set_dOut_dNet()
        self.set_delta(dETotal_dOut)
        self.set_dW()
        # update the total derivative error before weight updates
        self.set_dETotal_dOut()

        self.update(lr)


class NN:
    def __init__(self, loss, lr=0.01):
        self.clear_layers()
        self.lr = lr
        self.loss = loss_functions.__dict__[loss]
        self.loss_derivative = loss_functions.__dict__[loss+"_derivative_chain"]

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

    def backpropagate(self, Y, lr=.5):
        l_sz = len(self.layers)
        l_idx = l_sz - 1
        layer = self.layers[l_idx]
        dE_dW = self.loss_derivative(layer.out, Y)
        layer.backpropagate(dETotal_dOut=dE_dW, lr=lr)

        while l_idx > 0:
            l_idx -= 1
            layer = self.layers[l_idx]
            layer.backpropagate(output_layer=self.layers[l_idx+1])

    def fit(self, X, Y, lr=0.5,
            max_iter=1000, lr_optimizer=None,
            epsilon=0.001, power_t=0.25, t=1.0,
            batch_type='Full',
            batch_sz=1,
            print_interval=100,
            X_val=None,
            y_val=None):

        error = 1
        it = 0
        epoch = 0
        lst_epoch = 0
        b_it = 0
        nsamples = X.shape[0]

        self.lr = lr

        if batch_type == 'Full':
            b_sz = nsamples
        else:  # Mini or Stochastic
            b_sz = batch_sz

        if batch_type == 'Stochastic':
            X, Y = shuffle(X, Y)
            print('Shuffled')

        global __iteration_log
        __iteration_log = []
        error_val = 0.

        while ((error > epsilon) and (it < max_iter)):
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
                    if batch_type == 'Stochastic':
                        X, Y = shuffle(X, Y)

            # Only exits the loop when there is data for the batch
            
            _, aY = self.feed_forward(X_)

            error = self.loss(aY, Y_)

            self.backpropagate(Y_, lr)

            it += 1
            t += 1

            if X_val is not None:
                _, y_pred_val = self.feed_forward(X_val)
                error_val = ((y_val - y_pred_val) ** 2).mean() / 2

            if (it % print_interval) == 0 or it == 1:
                if X_val is not None:
                    print("It: %s Batch: %s Epoch %i Train Loss: %.8f lr: %.8f Val Loss: %.8f" %
                            (it, b_it, epoch, error, eta, error_val))
                else:
                    print("It: %s Batch: %s Epoch %i Error: %.8f lr: %.8f " %
                            (it, b_it, epoch, error, eta))

            if X_val is not None:
                __iteration_log.append(
                    (it, b_it, epoch, error, eta, error_val))
            else:
                __iteration_log.append((it, b_it, epoch, error, eta))

        if X_val is not None:
            print("Finished \n It: %s Batch: %s Epoch %i Train Loss: %.8f lr: %.8f Val Loss: %.8f" %
                    (it, b_it, epoch, error, eta, error_val))
            __iteration_log.append(
                (it, b_it, epoch, error, eta, error_val))
        else:
            print("Finished \n It: %s Batch: %s Epoch %i Train Loss: %.8f lr: %.8f " %
                    (it, b_it, epoch, error, eta))
            __iteration_log.append((it, b_it, epoch, error, eta))
