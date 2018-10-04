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
from utils import dataset_helper, custom_scores

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
        self.input_sz = input_sz
        self.neurons = neurons
        self.label = label

        self.model = None

        self.activation = activation_functions.__dict__[activation]

        self.act_derivative = activation_functions.__dict__[
            activation+"_derivative_chain"]

        if weights is not None:
            self.weights = weights
        else:
            self.weights = np.random.uniform(-1.0/math.sqrt(neurons), 1.0/math.sqrt(neurons),
                                             (input_sz, neurons))

        self.bias = np.zeros((1, neurons))
        if bias != .0:
            self.bias = bias

    def set_model(self, model):
        self.model = model

    def __str__(self):
        return inspect.cleandoc("""
        {}\t(input={}, neurons={}, activation={})""".format(self.label, self.input_sz, self.neurons, self.activation.__name__))

    def feed_forward(self, input_values):

        self.input = input_values
        self.net = input_values.dot(self.weights) + self.bias
        self.out = self.activation(self.net)
        return self.net, self.out

    def backpropagate(self, last_layer=None, output=False, loss_gradient=None, lr=.5):
        if output:
            # print(loss_gradient.mean())
            self.delta = loss_gradient * \
                self.act_derivative(self.net)
            self.grad_w = self.input.T.dot(self.delta)
            self.grad_bias = np.sum(self.delta, axis=0, keepdims=True)
            # print(self.label, 
            #     'delta', self.grad_w.mean(), 
            #     'grad', self.delta.mean(), 
            #     'act ', self.act_derivative(self.net).mean(),
            #     'net ', self.net.mean()
            #     )

        else:
            self.delta = last_layer.delta.dot(
                last_layer.weights.T) * self.act_derivative(self.net)
            self.grad_w = self.input.T.dot(self.delta)
            self.grad_bias = np.sum(self.delta, axis=0, keepdims=True)
            # print(self.label, 
            #     'delta', self.grad_w.mean(), 
            #     'grad', self.delta.mean(), 
            #     'act ', self.act_derivative(self.net).mean(),
            #     'net ', self.net.mean()
            #     )

    def update(self, lr=0.5):
        # print(self.label, self.weights.shape, self.grad_w.shape)
        self.weights -= lr * self.grad_w
        self.bias -= lr * self.grad_bias
        return 0


class NN:
    def __init__(self, loss):
        self.clear_layers()        
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
            layer.feed_forward(input_values)
            input_values = layer.out

        last_layer = self.layers[-1]
        return last_layer.net, last_layer.out

    def predict(self, X):
        input_values = X.copy()
        for layer in self.layers:
            hidden_input = input_values.dot(layer.weights) + layer.bias
            hidden_output = layer.activation(hidden_input)
            input_values = hidden_output
        return activation_functions.softmax(hidden_input)

    def backpropagate(self, Y, y_pred,lr=.5):
        l_sz = len(self.layers)
        l_idx = l_sz - 1
        layer = self.layers[l_idx]

        loss_derivative = self.loss_derivative(p=y_pred, y=Y)

        layer.backpropagate(last_layer=None, output=True,
                            loss_gradient=loss_derivative, lr=lr)
        while (l_idx > 0):
            l_idx -= 1
            layer = self.layers[l_idx]
            last_layer = self.layers[l_idx+1]
            layer.backpropagate(last_layer=last_layer, output=False,
                                loss_gradient=None, lr=lr)
            # print(layer.label, layer.grad_w.mean())

        for layer in self.layers:
            # print(layer.label, layer.grad_w.mean())
            layer.update(lr=lr)

    def fit(self, X, Y, lr=0.5,
            max_iter=1000, lr_optimizer=None,
            epsilon=0.001, power_t=0.25, t=1.0,
            print_interval=100, b_sz=1,
            decay_iteractions=None, decay_rate=None,
            X_val=None,
            Y_val=None):

        error = 1
        it = 0
        epoch = 0
        
        b_it = 0

        self.lr = lr

        X, Y = shuffle(X, Y)
        print('Shuffled')

        global __iteration_log
        __iteration_log = []
        error_val = 0.
        _error = 999999
        lst_epoch = 0.
        last_error = 0.
        while ( error > epsilon) and (it < max_iter):
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
            # X_, Y_ = X, Y
            # Only exits the loop when there is data for the batch

            # print(X_.shape)

            _, aY = self.feed_forward(X_)

            _error = self.loss(aY, Y_).mean()

            # print(aY - )
            # print('derivative',
            #       loss_functions.cross_entropy_derivative_chain(self.layers[-1].out, Y_).mean())
            # print('derivative', self.loss_derivative(Y_, aY).mean())

            error += _error

            self.backpropagate(Y=Y_, y_pred=aY,lr=eta)

            if _error < epsilon:
                last_error = _error 

            it += 1
            t += 1

            if (it % print_interval) == 0 and it > 1:

                error /= print_interval

                if X_val is not None:
                    y_pred_val = np.array(self.predict(X_val))
                    error_val = self.loss(y_pred_val, Y_val).mean()
                    val_acc = custom_scores.accuracy_score(Y_val.argmax(
                        axis=-1), y_pred_val.argmax(axis=-1), mode='multi')
                if X_val is not None:
                    print("It: %s Batch: %s Epoch %i Train Loss: %.8f lr: %.6f Val Loss: %.8f Val Acc %.8f" %
                          (it, b_it, epoch, error, eta, error_val, val_acc))
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
