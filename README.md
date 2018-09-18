# LogisticRegression-ANN

## Introduction

Using fashion-mnist as dataset we are supposed to implement from scracth an Artificial Neural Network with an experimental protocol pre-defined.

The main activites to be delivered were stated by our professor as follows:

1. Perform Logistic Regression as the baseline (first solution) to learn the 10 classes in the dataset. Use one-vs-all strategy to build a classification model. Keep in mind that you should obtain 10 classification models.
2. Perform Multinomial Logistic Regression (i.e., Softmax regression). It is a generalization of Logistic Regression to the case where we want to handle multiple classes. What are the conclusions?
3. Move on to Neural Networks, using one hidden layer. You should implement your solution.
4. Extend your Neural Network to two hidden layers. Try different activation functions. Does the performance improve?
5. Pick your best model and plot the confusion matrix in the test set. What are the conclusions?
6. Prepare a 4-page (max.) report with all your findings. It is UP TO YOU to convince the reader that you are proficient on Logistic Regression and Neural Network, and the choices it entails.

## Planning

1. Performe an EDA over the data to understand its values, classes and distributions.
    * Reserve the test data.
    * Identify metrics for each method (Logistic Regression one-vs-all, Multinomial Logistic Regression and Neural networks.)
    * Data cleaning (negative values removal or invalid images).
2. Perform the logistic regression as told.
    * Use default parameters to define the benchmark (The online value for it is of 84% aprox.) `http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/#`.
    * Perform a grid search for the logistic regression and also identify augmentation or data cleaning methods if needed.
