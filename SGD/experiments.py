import math
import os
import sys
sys.path.append("./")
import time
import timeit

from SGD.custom_SGD import *
from SGD import softmax_logistic

import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from utils import custom_scores, dataset_helper


#
#  This method performs a k-fold cross validation in order to define the
# best hyperparameters for the linear regression model
#  
# params:
#   X_train       -> training features
#   y_train       -> training target
#   nfolds        -> number of foldes for the k-fold method
#   learning_rate -> set of learning rates candidates (list)
#   iterations    -> set of iterations candidates (list)
#
# return:
#   set of best parameters
#
def Kfold(X,y,k):

	nsamples = X.shape[0]
	nfeatures = X.shape[1]

	X = np.insert(X,nfeatures,y,axis=1)
	nfeatures += 1
	
	np.random.shuffle(X)
	nfold = nsamples // k
	rest = nsamples % k

	X = X.reshape((k,nfold,nfeatures))

	if (rest != 0):
		Xrest = X[-rest:]
		X = X[:-rest]	
		j = 0
		for x in Xrest:
			X[j].append(x)
			j += 1

	return X


#
#  This method separates the K-fold into train and test folds
#  
# params:   
#   folds -> folds containing the training sets divided. Shape: (k,nsamples/k,nfeatures)
#   i     -> index of the fold to be used as test
#
# return:
#   fold_train -> fold of training samples
#   fold_test  -> fold of test samples
#
def separateTrainTestFolds(folds,i):
    fold_train = np.concatenate([folds[:i],folds[i+1:]])
    fold_test = folds[i]
    fold_X_train = fold_train.reshape((fold_train.shape[1]*fold_train.shape[0],fold_train.shape[2]))
    fold_y_train = fold_X_train[:,fold_X_train.shape[1]-1]
    fold_X_test = fold_test.copy()
    fold_y_test = fold_X_test[:,fold_X_test.shape[1]-1]
    fold_X_train = np.delete(fold_X_train,fold_X_train.shape[1]-1,axis=1)
    fold_X_test = np.delete(fold_X_test,fold_X_test.shape[1]-1,axis=1)
    return fold_X_train, fold_y_train, fold_X_test, fold_y_test


#
#  This method performs a k-fold cross validation in order to define the
# best hyperparameters for the linear regression model
#  
# params:
#   X_train       -> training features
#   y_train       -> training target
#   nfolds        -> number of foldes for the k-fold method
#   learning_rate -> set of learning rates candidates (list)
#   iterations    -> set of iterations candidates (list)
#
# return:
#   set of best parameters
#
def crossValidation(X_train,y_train,nfolds,params,classifier='one_vs_all'):

    nclasses = len(np.unique(y_train))

    train_nclasses = 0

    while (nclasses != train_nclasses):

        folds = Kfold(X_train,y_train,nfolds)
        for f in range(nfolds):
            fold_X_train, fold_y_train, fold_X_val, fold_y_val = separateTrainTestFolds(folds,f)	
            train_nclasses = len(np.unique(fold_y_train))
            if (nclasses != train_nclasses):
                break

    print("======================================")
    print(" K-Fold Cross Validation Grid Search")
    print(" Params:")
    for i in params:
        print(i)
    print("======================================")

    score = {}
    print_interval = 100
    batch_sz = 64

    for i in params:
        acc = []
        pre = []
        recall = []
        fb = []
        err = []

        lr = i['lr']
        it = i['max_iter']
        print("Testing params:")
        print("lr =",lr)
        print("it =",it)

        for f in range(nfolds):
            print("----- Fold",f)
            fold_X_train, fold_y_train, fold_X_val, fold_y_val = separateTrainTestFolds(folds,f)

            if (classifier == 'one_vs_all'):

                theta = SGD_one_vs_all(**i, X=fold_X_train, y=fold_y_train, 
                               batch_sz=1, print_interval=print_interval)

                y_pred = classify(theta, fold_X_val, binary=False)

            if (classifier == 'softmax'):
                
                theta,_ = softmax_logistic.BGD(X = fold_X_train, y = fold_y_train,**i, print_interval=print_interval)

                y_pred = softmax_logistic.classify_softmax(theta,fold_X_val)

            acc.append(custom_scores.accuracy_score(fold_y_val, y_pred, mode='multi'))
            pre.append(custom_scores.precision_score(fold_y_val, y_pred, mode='multi'))
            recall.append(custom_scores.recall_score(fold_y_val, y_pred, mode='multi'))
            fb.append(custom_scores.f1_score(fold_y_val, y_pred, mode='multi'))

        err.append([np.mean(acc),np.std(acc)])
        err.append([np.mean(pre),np.std(pre)])
        err.append([np.mean(recall),np.std(recall)])
        err.append([np.mean(fb),np.std(fb)])

        score[lr,it] = []
        err = np.array(err).reshape((4,2)) # nmetrics (4) plus mean and std (2)		
        score[lr,it].append(np.mean(err,axis=0))

    print("------ Final Scores!")	
    for i in params:
        print(i)
        print(score[i['learning_rate'],i['iterations']])
        
    return score