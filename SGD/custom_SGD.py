import math
import os
import sys 
sys.path.append("../")
import time
import timeit

import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from numpy.core.umath_tests import inner1d
from pandas.api.types import CategoricalDtype
from sklearn import ensemble, linear_model, metrics, model_selection
from sklearn.datasets import load_breast_cancer, make_regression
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from utils import custom_scores



__iteration_log = []


def get_iteration_log():
    global __iteration_log
    cols = len(__iteration_log[0])
    print(cols)
    df = pd.DataFrame(__iteration_log, columns=['it', 'b_it', 'epoch', 'error_train', 'eta', 'error_val'][:cols])
    df.set_index(df.it)
    return df


def get_batch(X, y, b_it, b_sz, epoch):
    b_ct = int(X.shape[0]/b_sz)
    y_ = np.zeros((0, 0))
    X_ = np.zeros((0, 0))
    start = 0
    finish = 0

    if b_it > b_ct:
        b_it = 0
        epoch += 1
    start = b_it * b_sz
    finish = (b_it+1) * b_sz
    X_ = X[start: finish]
    y_ = y[start: finish]

    b_it += 1

    return X_, y_, b_it, epoch


def get_batch_test():
    count = 105
    X = np.ones((count, 3))
    y = np.array(range(0, count))
    b_it = 0
    b_sz = 100
    epoch = 0
    it = 0
    sz = 0
    while epoch < 1:
        X_, y_, b_it, epoch = get_batch(X, y, b_it, b_sz, epoch)
        sz += X_.shape[0]
        print(it, sz, X_.shape, y_.shape, b_it, b_sz, epoch)
        print(y_)
        it += 1


'''
 Since we are dealing with logistic regression,
 the hypothesis is defined as:

                    1
       F(x) = ----------------
                1 + exp^(-x)

 However, its implementation may result in overflow
 if x is too large, then, the version implemented 
 here is more stable with similar results, and is
 defined as:
 
                  exp^(x)
       F(x) = ----------------, if x < 0
                1 + exp^(x) 
                
                    1
       F(x) = ----------------, if x >= 0
                1 + exp^(-x) 
'''
def hypothesis(theta,X,stable=False):
    
    dot = np.dot(X,theta)
    
    #Regular Sigmoid Function        
    if (stable == False):        
        h = 1 / (1 + np.exp(-dot))
    
    else:
    #Stable Sigmoid Function
        num = (dot >= 0).astype(np.float128)
        dot[dot >= 0] = -dot[dot >= 0]	
        exp = np.exp(dot)
        num = np.multiply(num,exp)
        h = num / (1 + exp)
    
    return h



# Given a threshold apply a 
# binary classification of the samples
# regarding an optimized theta
def classify(theta, X, th):
    X = np.insert(X, 0, 1, axis=1)
    y = hypothesis(theta, X)
    y[y >= th] = 1
    y[y < th] = 0
    X = np.delete(X, 0, axis=1)
    return y


# Apply a multi class classification of the samples
# regarding an optimized set of thetas
def classify_multiclass(theta , X):
	X = np.insert(X,0,1,axis=1)
	classes = []
	max_prob = np.array([])
	for m in theta:			
		h = hypothesis(theta[m],X)
		if max_prob.size == 0:
			max_prob = h
			classes = [m]*h.shape[0]
		for i in range(len(h)):
			if h[i] > max_prob[i]:
				max_prob[i] = h[i]
				classes[i] = m
	X = np.delete(X,0,axis=1)
	return classes


def cross_entropy_loss(h, y):
    # y.log(h) + (1-log(h) . 1-y)
    # log probability * inverse of the log probabality 
	eps = np.finfo(np.float).eps
	h[h < eps] = eps
	h[h > 1.-eps] = 1.-eps
	return np.multiply(np.log(h),y) + np.multiply((np.log(1-h)),(1-y))

def grad_logit_step(theta, X, y, alpha, error):
    """
    Given the current Theta Set it calculates the gradient and new values for it.
    """
    grad = np.dot(X.transpose(),error)/len(y)
    result = theta - alpha * grad

    return result


def grad_logit_step_test():
    theta = np.array([1, 0, 0], dtype='float64')
    theta_temp = np.array([0, 0, 0], dtype='float64')
    X, y = get_toy_data()
    X = np.insert(X, 0, 1, axis=1)

    print("X values ")
    print(X)
    alpha = .9
    max_iter = 50
    for i in range(max_iter):
        h0 = hypothesis(theta, X)
        error = (h0 - y)
        # for j in range(X.shape[1]):
        theta_temp = grad_logit_step(theta, X, y, alpha, error)

        theta = theta_temp.copy()
        print("Iter %i theta: %s" % (i, theta))
        y_hat = hypothesis(theta, X)
        error = math.sqrt(((y_hat-y)**2).mean())
        print("RMSE error: %.4f" % error)

    theta = np.array(theta)
    print("Predicted: %s" % (hypothesis(theta, X)))
    print("Expected: %s" % (y))


def get_toy_data():
    y = np.array([1., 0.], dtype='float64')
    X = np.array([[4., 7.], [2., 6.]], dtype='float64')
    return X, y


def get_toy_data_big():
    """
        Returns  X_train, X_test, y_train, y_test from Breat Cancer
    """
    X,y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    return X_train, X_test, y_train, y_test

def SGD(lr, max_iter, X, y, lr_optimizer=None,
        epsilon=0.001, power_t=0.25, t=1.0,
        batch_type='Full',
        batch_sz=1,
        print_interval=100,
        X_val=None,
        y_val=None):

    # Adding theta0 to the feature vector
    X = np.insert(X, values=1, obj=0, axis=1)

    shape = X.shape
    nsamples = shape[0]
    print("Number of samples: "+str(nsamples))
    nparams = shape[1]
    print("Number of parameters: "+str(nparams))

    theta = np.zeros(nparams)
    theta_temp = np.ones(nparams)

    error = 1
    it = 0
    epoch = 0
    lst_epoch = 0
    b_it = 0

    if batch_type == 'Full':
        b_sz = nsamples
    else:  # Mini or Stochastic
        b_sz = batch_sz

    if batch_type == 'Stochastic':
        X, y = shuffle(X, y)
        print('Shuffled')
    global __iteration_log
    __iteration_log = []
    error_val = 0.
    while ( ( (epsilon is None) or (error > epsilon) ) and (it < max_iter) ):
        if lr_optimizer == 'invscaling':
            eta = lr / (it + 1) * pow(t, power_t)
        else:
            eta = lr

        X_ = np.zeros(0)
        y_ = np.zeros(0)
        while y_.shape[0] == 0:
            # Checking if it is a new epoch to shuffle the data.
            X_, y_, b_it, epoch = get_batch(X, y, b_it, b_sz, epoch)
            if lst_epoch < epoch:
                lst_epoch = epoch
                if batch_type == 'Stochastic':
                    X, y = shuffle(X, y)

        h0 = hypothesis(theta, X_)

        error = (h0 - y_)

        theta_temp = grad_logit_step(theta, X_, y_, eta, error)

        y_pred = hypothesis(theta_temp, X_)
        error = ((y_ - y_pred) ** 2).mean() / 2

        theta = theta_temp.copy()

        it += 1
        t += 1

        if X_val is not None:
            y_pred_val = predict(theta, X_val)
            error_val = ((y_val - y_pred_val) ** 2).mean() / 2

        if (it % print_interval) == 0 or it == 1:
            if X_val is not None:
                print("It: %s Batch: %s Epoch %i Train Loss: %.8f lr: %.8f Val Loss: %.8f" %
                      (it, b_it, epoch, error, eta, error_val))
            else:
                print("It: %s Batch: %s Epoch %i Error: %.8f lr: %.8f " %
                      (it, b_it, epoch, error, eta))

        if X_val is not None:
            __iteration_log.append((it, b_it, epoch, error, eta, error_val))
        else:
            __iteration_log.append((it, b_it, epoch, error, eta))

    if X_val is not None:
        print("Finished \n It: %s Batch: %s Epoch %i Train Loss: %.8f lr: %.8f Val Loss: %.8f" %
              (it, b_it, epoch, error, eta, error_val))
        __iteration_log.append((it, b_it, epoch, error, eta, error_val))
    else:
        print("Finished \n It: %s Batch: %s Epoch %i Train Loss: %.8f lr: %.8f " %
              (it, b_it, epoch, error, eta))
        __iteration_log.append((it, b_it, epoch, error, eta))
    return theta



def predict(theta,X):
    X = np.insert(X, 0, 1, axis=1)
    y = hypothesis(theta, X)
    X = np.delete(X, 0, axis=1)
    return y


def SGD_test():    
    X,  X_val, y, y_val = get_toy_data_big()        
    print("X values ")
    print(X)
    lr = .01
    max_iter = 10000
    batch_sz = 100
    print_interval=1000

    print("")
    print("Full batch")

    start = time.process_time()
    theta = SGD(lr, max_iter, X, y, batch_type='Full', print_interval=print_interval)
    print("finished ", time.process_time() - start)
    y_pred = predict(theta, X_val)
    print("Accuracy: %.3f" % custom_scores.accuracy_score(y_val, y_pred))
    print("Precision: %.3f" % custom_scores.precision_score(y_val, y_pred))
    print('Recall: %.3f' % custom_scores.recall_score(y_val, y_pred))

    print("")
    print("Stochastic Mini batch")
    start = time.process_time()
    theta = SGD(lr, max_iter, X, y, batch_type='Stochastic',
                batch_sz=batch_sz, print_interval=print_interval)
    print("finished ", time.process_time() - start)
    y_pred = predict(theta, X_val)
    print("Accuracy: %.3f" % custom_scores.accuracy_score(y_val, y_pred))
    print("Precision: %.3f" % custom_scores.precision_score(y_val, y_pred))
    print('Recall: %.3f' % custom_scores.recall_score(y_val, y_pred))

    print("")
    print("Mini batch")
    start = time.process_time()
    theta = SGD(lr, max_iter, X, y, batch_type='Mini',
                batch_sz=batch_sz, print_interval=print_interval)
    print("finished ", time.process_time() - start)
    y_pred = predict(theta, X_val)
    print("Accuracy: %.3f" % custom_scores.accuracy_score(y_val, y_pred))
    print("Precision: %.3f" % custom_scores.precision_score(y_val, y_pred))
    print('Recall: %.3f' % custom_scores.recall_score(y_val, y_pred))

    print("")
    print("Single Instance")
    start = time.process_time()
    theta = SGD(lr, max_iter, X, y, batch_type='Single',
                epsilon=None, batch_sz=1, print_interval=print_interval)
    print("finished ", time.process_time() - start)
    y_pred = predict(theta, X_val)
    print(y_pred.shape, y_val.shape)
    print("Accuracy: %.3f" % custom_scores.accuracy_score(y_val, y_pred))
    print("Precision: %.3f" % custom_scores.precision_score(y_val, y_pred))
    print('Recall: %.3f' % custom_scores.recall_score(y_val, y_pred))
