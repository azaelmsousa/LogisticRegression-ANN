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
        self.output_layer = w[-1]class Function:
	
	#
	# Sigmoid function for the activation
	# of a neuron, where h is the dot
	# product of X (input) and theta (weights)
	#
	def act_Sigmoid(h):
		sig = 1. / (1. + np.exp(-h))
		return sig

	#
	# Derivative of the sigmoid function. It
	# is used as part of the backpropagation
	# algorithm
	#
	def act_Sigmoid_derivative(h):
		sig = act_Sigmoid(h)
		derivative = sig*(1-sig)
		return derivative

	#
	# Hyberbolic tangent function for the
	# acivation of a neuron, where h is the
	# dot product of X (input) and theta (weights)
	#	
	def act_Tanh(h):
		tanh = (2 / (1+np.exp(-2*h)))-1

	#
	# Deivative of the hyperbolic tangent
	# function. It is used as part of the	
	# back propagation algorithm.
	#
	def act_Tanh_derivative(h):
		tanh_l = (4*np.exp(-2*h))/((1+np.exp(-2*h))**2)
		return tanh_l

	#
	# Cross entropy loss function, where h
	# is the activation of the last layer.
	# It computes the error of the predicted
	# class and the correct one.
	#
	def err_CrossEntropy(predY,y):
		eps = np.finfo(np.float128).eps
		predY[predY < eps] = eps
		predY[predY > 1.-eps] = 1.-eps
		return -np.multiply(np.log(predY),y) - np.multiply((np.log(1-predY)),(1-y))

	#
	# Derivative of the SMD loss function.
	# It is used in the back propagation
	# algorithm.
	#
	def err_CrossEntropy_derivative(X,predY,y):
		error = (predY - y)
		grad = np.dot(X.transpose(),error)
		return grad

	#
	# Sum of the squared differences (SMD) loss
	# function, where h is the activation of the
	# last layer. It computes the error of the
	# predicted class and the correct one.
	#
	def err_SMD(predY,y):
		error = np.square((predY - y)).sum()
		return error/2

	#
	# Sum of the squared differences (SMD) loss
	# function, where h is the activation of the
	# last layer. It computes the error of the
	# predicted class and the correct one.
	#
	def err_SMD_derivative(X,predY,y):
		error = (predY - y)
		grad = np.dot(X.transpose(),error)
		return grad


class ANN:

	def __init__(self, activation, error):
		self.act_func = Function.__dict__[activation]
		self.act_derivative = Function.__dict__[activation+"_derivative"]
		self.err_func = Function.__dict__[error]
		self.err_derivative = Function.__dict__[error+"_derivative"]
		self.activations = []
		self.dot_product = []

	def initialize_random_weights(self, n_input, n_perceptron, n_classes):
		self.n_hidden_layers = len(n_perceptron)

		self.hidden_layers = []		

		for l in range(self.n_hidden_layers):
			if (l == 0):
				w = np.random.rand(n_input+1,n_perceptron[l])				
			else:
				w = np.random.rand(n_perceptron[l-1]+1,n_perceptron[l])
			self.hidden_layers.append(w)
			

		if (self.n_hidden_layers == 0):
			self.output_layer = np.random.rand(n_input+1,n_classes)
		else:
			self.output_layer = np.random.rand(n_perceptron[self.n_hidden_layers-1]+1,n_classes)


	def initialize_fixed_weights(self, w):
		self.hidden_layers = w[:-1]
		self.output_layer = w[-1]
		self.n_hidden_layers = len(w)-1


	def show_weights(self):

		for l in range(self.n_hidden_layers):
			print("Hidden Layer ",str(l+1))
			print(self.hidden_layers[l],"\n")

		print("Output Layer ")
		print(self.output_layer,"\n")

	def show_activations(self):

		for l in range(self.n_hidden_layers):
			print("Hidden Layer ",str(l+1))
			print(self.activations[l],"\n")

		print("Output Layer ")
		print(self.activations[self.n_hidden_layers],"\n")

	def show_setup(self):
				
		print("--- Input size: ",str(self.hidden_layers[0].shape[0]-1))
		print("--- Number of hidden layers: ",str(self.n_hidden_layers))
		print("--- Number of perceptrons at each layer: ")
		for l in range(self.n_hidden_layers):
			print("------ HL "+str(l+1)+": "+str(self.hidden_layers[l].shape[1]))
		print("--- Number of classes: "+str(self.output_layer.shape[1]),"\n")

	def foward_propagation(self, X):

		del self.activations[:]

		self.activations.append(X)
		inp = np.insert(X,0,1,axis=1)

		for l in range(self.n_hidden_layers):
			out = np.matmul(inp, self.hidden_layers[l])
			self.dot_product.append(out)
			sig = self.act_func(out)
			self.activations.append(sig)
			inp = np.insert(sig,0,1,axis=1)			

		out = np.matmul(inp, self.output_layer)
		self.dot_product.append(out)
		sig = self.act_func(out)
		self.activations.append(sig)

		return sig

	def backpropagation(self, x, y):
		sig = self.foward_propagation([x])
		#dErr/dAct * dAct/dDot
		delta1 = self.err_derivative(sig,y) * self.act_derivative(self.dot_product[-1])	
	
		#input current layer = output previous layer (inserting bias)
		act = np.insert(self.activations[-2],0,1,axis=1).transpose()

		#dErr/dAct * dAct/dDot * dDot/dWl
		grad1 = np.matmul(act,delta1)

		#Layer (l-1)

		#dDotK/dAct (no bias)
		dot_act = self.output_layer[1:]
		#dAct/dDot
		act_dot = self.act_derivative(self.dot_product[-2])
		delta2 = np.multiply(dot_act, act_dot.transpose())
		delta12 = np.multiply(delta1,delta2)
		delta12s = delta12.sum(axis=1)
		delta12s = np.expand_dims(delta12s, axis=0)

		act = np.insert(self.activations[-3],0,1,axis=1).transpose()
		grad2 = np.multiply(delta12s, act)

		grad3 = None
		#Layer (l-2)
		if(len(self.hidden_layers) > 1):
			#dDotK/dAct (no bias)
			dot_act = self.hidden_layers[-1][1:]
			#dAct/dDot
			act_dot = self.act_derivative(self.dot_product[-3])
			delta3 = np.multiply(dot_act, act_dot.transpose())		

			delta123 = np.array([])
			for l,c in zip(delta12,delta3.transpose()):
				l = np.expand_dims(l,axis=0)
				c = np.expand_dims(c,axis=1)
				matrix = np.matmul(c,l)
				delta123 = delta123+matrix if delta123.size else matrix
			delta123s = delta123.sum(axis=1)

			act = np.insert([x],0,1,axis=1).transpose()
			grad3 = np.multiply(delta123s,act)
		
		return grad1,grad2,grad3


	def stochastic_training(self, X, Y, alpha, epochs):
		J = []
		for e in range(epochs):
			for x,y in zip(X,Y):
				grad1, grad2, grad3 = self.backpropagation(x,y)			
			
				#update weights
				self.output_layer      = self.output_layer      - alpha*grad1
				self.hidden_layers[-1] = self.hidden_layers[-1] - alpha*grad2
				if(len(self.hidden_layers) > 1): self.hidden_layers[-2] = self.hidden_layers[-2] - alpha*grad3

				sig = self.foward_propagation([x])
				err = self.err_func(sig,y)
				J.append(err)
				
		plt.plot(J)	
		plt.ylabel('Error')
		plt.xlabel('iterations')
		plt.show()

	def batch_training(self, X, Y, alpha, epochs):
		J = []
		for e in range(epochs):
			grad1s, grad2s, grad3s = np.array([]),np.array([]),np.array([])
			for x,y in zip(X,Y):
				grad1, grad2, grad3 = self.backpropagation(x,y)			
				grad1s = grad1s+grad1 if grad1s.size else grad1
				grad2s = grad2s+grad2 if grad2s.size else grad2
				if grad3: grad3s = grad3s+grad3 if grad3s.size else grad3

			grad1s = grad1s/X.shape[0]
			grad2s = grad2s/X.shape[0]
			grad3s = grad3s/X.shape[0]
			
			#update weights
			self.output_layer      = self.output_layer      - alpha*grad1s
			self.hidden_layers[-1] = self.hidden_layers[-1] - alpha*grad2s
			if(len(self.hidden_layers) > 1): self.hidden_layers[-2] = self.hidden_layers[-2] - alpha*grad3s

			sig = self.foward_propagation(X)
			err = self.err_func(sig,Y)
			J.append(err.sum()/X.shape[0])
				
		plt.plot(J)	
		plt.ylabel('Error')
		plt.xlabel('iterations')
		plt.show()
