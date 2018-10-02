# LogisticRegression-ANN

## Introduction

Using fashion-mnist as dataset we are supposed to implement from scracth an Artificial Neural Network with an experimental protocol pre-defined.

The main activites to be delivered were stated by our professor as follows:

### Activities Needed

1. Perform Logistic Regression as the baseline (first solution) to learn the 10 classes in the dataset. Use one-vs-all strategy to build a classification model. Keep in mind that you should obtain 10 classification models.
2. Perform Multinomial Logistic Regression (i.e., Softmax regression). It is a generalization of Logistic Regression to the case where we want to handle multiple lasses. What are the conclusions?
3. Move on to Neural Networks, using one hidden layer. You should implement your solution.
4. Extend your Neural Network to two hidden layers. Try different activation functions. Does the performance improve?
5. Pick your best model and plot the confusion matrix in the test set. What are the conclusions?
6. Prepare a 4-page (max.) report with all your findings. It is UP TO YOU to convince the reader that you are proficient on Logistic Regression and Neural Network, and the choices it entails.

### Code Structure

- `NN` Module containing the Neural Network implementation alongside with its helpers.
- `data` Directory containing the target dataset `Fasion Mnist`.
- `notebooks` The experiments and tests for the code are all builtin upon using jupyter notebooks in this directory.
- `SGD` This module has the implementation of the Logistic Regresion and the Multinomial Logistic Regression as asked.
- `helper` In the module there helper functions to read the mnist data format, work with datasets and evaluate the algorithms.

## Planning

1. Performe an EDA over the data to understand its values, classes and distributions.
   - Reserve the test data. (Done using the `EDA Fashion Mnist` notebook)
   - Identify metrics for each method (Logistic Regression one-vs-all, Multinomial Logistic Regression and Neural networks.) (Done using the `SGD Logistic Test` notebook)
   - Data cleaning (negative values removal or invalid images). (As shown by the `EDA Fashion Mnist` notebook, the dataset appears to be well-behaved.)
2. Perform the logistic regression (One vs All) on the dataset :
   - Implement and test the logistic regression using a binary dataset (make_classifer library from sklearn might help). (Done using the `SGD Logistic Test` notebook)
   - Implement and test the "One vs All" Strategy for classification (make_classifer library from sklearn might help). (Done using the `SGD Logistic Test` notebook)
   - Use default parameters to define the benchmark (The online value for it is of 84% aprox.) `http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/#`.
   - Perform a grid search for the logistic regression and also identify augmentation or data cleaning methods if needed.
   - Only using the Training Set.
3. Considering the following workflow perform the Neural Network trainning over the dataset.

   - Implement a 1 hidden layer neural network. (Done using the `ANN_Test` notebook)
     - Considering pseudo algorithm for the neural network given in the class slides as test set.
     - Loss functions: cross entropy, sum of squares.
     - Activation Functions: Tahn, Logistic (sigmoid) and Softmax.
   - Train with a network with one hidden layer and evaluate with the Training set.
   - Grid search for the network architecture:
     - Number Layer.
     - Number of Neurons per layer.
   - Only using the Training Set.
4. Compare the models using the Test Set.