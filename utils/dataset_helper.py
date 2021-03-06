import numpy as np
from sklearn import ensemble, linear_model, metrics, model_selection
from sklearn.datasets import load_breast_cancer, make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from . import mnist_reader

__RANDOM_STATE = 42


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


def one_hot_encode(Y, nclasses):
    y = Y.copy().reshape(-1)
    return np.eye(nclasses)[y]


def one_hot_decode(Y):
    return Y.argmax(axis=-1)


def get_toy_data_multiclass(nclasses=4, nsamples=500, nfeatures=10):
    """
        Returns  X_train, X_test, y_train, y_test from with 4 classes and 20 features
    """
    X, y = make_classification(n_samples=nsamples, n_features=nfeatures, n_classes=nclasses,
                               n_clusters_per_class=1, n_informative=4,
                               n_redundant=0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=__RANDOM_STATE)

    return X_train, X_test, y_train, y_test


def get_toy_data_binary():
    """
        Returns  X_train, X_test, y_train, y_test from Breast Cancer
    """
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=__RANDOM_STATE)
    return X_train, X_test, y_train, y_test


def get_toy_data():
    y = np.array([1., 0.], dtype='float64')
    X = np.array([[4., 7.], [2., 6.]], dtype='float64')
    return X, y


def load_fasion_mnist(base_dir='../data/fashion', scaling='default'):
    X_train, y_train = mnist_reader.load_mnist(base_dir, kind='train')
    X_test, y_test = mnist_reader.load_mnist(base_dir, kind='t10k')

    #
    # Normalizing values
    #
    if (scaling == 'default'):
        X_train = X_train / 255.
        X_test = X_test / 255.
    if (scaling == 'mean_std'):
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)
        X_train = (X_train - mean)/std
        X_test = (X_test - mean)/std
    if (scaling == 'min_max'):
        min_ = np.amin(X_train, axis=0)
        max_ = np.amax(X_train, axis=0)
        X_train = (X_train - min_)/(max_ - min_)
        X_test = (X_test - min_)/(max_ - min_)

    return X_train, y_train, X_test, y_test
