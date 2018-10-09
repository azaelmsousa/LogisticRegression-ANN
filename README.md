# LogisticRegression-ANN

## Introduction

Using Logistic Regression and Artificial Neural Networks, we aim to classify clothes in the fasion-MNIST dataset. All models and optimization methods employed in this project were implemented from scratch.

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
- `SGD` This module has the implementation of the Logistic Regresion and the Multinomial Logistic Regression.
- `helper` In the module there helper functions to read the mnist data format, work with datasets and evaluate the algorithms.

## Requirements

`$ pip install -r requirements.txt --user`

* backcall==0.1.0
* cycler==0.10.0
* decorator==4.3.0
* ipykernel==4.9.0
* ipython==6.5.0
* ipython-genutils==0.2.0
* jedi==0.12.1
* jupyter-client==5.2.3
* jupyter-core==4.4.0
* kiwisolver==1.0.1
* matplotlib==2.2.3
* numpy==1.15.1
* pandas==0.23.4
* parso==0.3.1
* pexpect==4.6.0
* pickleshare==0.7.4
* prompt-toolkit==1.0.15
* ptyprocess==0.6.0
* pycurl==7.43.0
* Pygments==2.2.0
* pygobject==3.20.0
* pyparsing==2.2.0
* python-apt==1.1.0b1
* python-dateutil==2.7.3
* pytz==2018.5
* pyzmq==17.1.2
* scikit-learn==0.19.2
* scipy==1.1.0
* simplegeneric==0.8.1
* six==1.11.0
* sklearn==0.0
* tornado==5.1
* traitlets==4.3.2
* wcwidth==0.1.7

## Experiments

### Logistic Regression

1) Compute the best hyperparameters (learning rate and number of iterations) using a K-Fold Cross Validation Grid Search, with k = 5.
2) Divide the Training set into Train and Validation using a constant random state (42) for future comparison.
3) Train the model and test is with the Validation set.

### Multinomial Logistic Regression

The experiments for this method is really similar to the previous one.

1) Compute the best hyperparameters (learning rate and number of iterations) using a K-Fold Cross Validation Grid Search, with k = 5.
2) Divide the Training set into Train and Validation using a constant random state (42) for future comparison.
3) Train the model and test is with the Validation set.

### ANN

1) ANN with 1 layer:
 * Change the loss function between sigmoid and ReLU
 * Test 2 number of neurons for each of them:
      * 256 neurons for layer 1
      * 128 neurons for layer 1

2) ANN with 2 layer:
 * Change the loss function between sigmoid and ReLU
 * Test 2 number of neurons for them:
      * 256 neurons at each layer
      * 128 neurons at each layer

The number of neurons were chosen as powers of 2 and less, but close, then the number of features.

### Final Experiment

The final experiment is done by selecting the model who performed best in the validation set. Then, this model is used in the test set. 

## Reproduction

To reproduce the results reported, run the following notebooks:

 * `EDA - Fashion Mnist.ipynb` This notebook contains the Exploratory Data Analisys, where the projection method t-SNE was applied over the dataset in order to give us a better understanding about the samples.
 * `Experiments - Round 1.ipynb` This notebook runs the Logistic Regression with the best hyperparameters defined at `CV - Round 1.ipynb`. Since the computation of the hyperparameters take a long time, we will not list it here to be executed, but it can be if the professor so desire.
 * `Experiments - Round 2.ipynb` This notebook runs the Multinomial Logistic Regression with the best hyperparameters defined at `CV - Round 2.ipynb`. The same restriction of the previous notebbok applies here.
 
 * `Neural Network Experiments - Round 1.ipynb`

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
     - Considering pseudo algorithm for the neural network given in the class slides as test set. (Done using the `ANN_Test` notebook)
     - Loss functions: cross entropy, sum of squares.  (Done using the `ANN_Test` notebook)
     - Activation Functions: Tahn, Logistic (sigmoid) and Softmax.  (Done using the `ANN_Test` notebook)
   - Train with a network with one hidden layer and evaluate with the Training set.  (Done using the `ANN_Test` notebook)
   - Grid search for the network architecture:
     - Number Layer.
     - Number of Neurons per layer.
   - Only using the Training Set.
4. Compare the models using the Test Set.
