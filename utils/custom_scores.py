import numpy as np
import matplotlib.pyplot as plt 
import itertools
from sklearn.metrics import confusion_matrix

def accuracy_score(Y, predY, mode='binary'):
    acc = 0.0
    if (mode == 'binary'):
        TP = ((predY == Y) & (predY == 1.)).sum()
        TN = ((predY == Y) & (predY == 0.)).sum()
        acc = (TP + TN) / Y.shape[0]
    elif (mode == 'multi'):
        TP = (predY == Y).sum()
        acc = TP / Y.shape[0]
    return acc


def precision_score(Y, predY, mode='binary'):
    precision = 0.0
    if (mode == 'binary'):
        TP = ((predY == Y) & (predY == 1)).sum()
        FP = ((predY != Y) & (predY == 1)).sum()
        precision = TP / (TP + FP)
    elif (mode == 'multi'):
        classes = np.unique(Y)
        for c in classes:
            TP = ((predY == Y) & (predY == c)).sum()
            FP = ((predY != Y) & (predY == c)).sum()
            precision += TP / (TP + FP)
        precision /= len(classes)
    return precision


def recall_score(Y, predY, mode='binary'):
    recall = 0.0
    if (mode == 'binary'):
        TP = ((predY == Y) & (predY == 1)).sum()
        FN = ((predY != Y) & (predY == 0)).sum()
        recall = TP / (TP + FN)
    elif (mode == 'multi'):
        classes = np.unique(Y)
        for c in classes:
            TP = ((predY == Y) & (predY == c)).sum()
            FN = ((predY != Y) & (Y == c)).sum()
            recall += TP / (TP + FN)
        recall /= len(classes)
    return recall


def f1_score(Y, predY, beta=1, mode='binary'):
    fscore = 0.0
    if (mode == 'binary'):
        precision = precision_score(predY, Y)
        recall = recall_score(predY, Y)
        fscore = (1 + beta*beta)*((precision*recall) /
                                  ((beta*beta*precision)+recall))
    elif (mode == 'multi'):
        precision = precision_score(predY, Y, 'multi')
        recall = recall_score(predY, Y, 'multi')
        fscore = (1 + beta*beta)*((precision*recall) /
                                  ((beta*beta*precision)+recall))
    return fscore


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def compute_confusion_matrix(y_test, y_pred, class_names=None):
    # Compute confusion matrix
    if class_names is None: 
        classes  = np.unique(y_test)
        class_names = {str(c):c for c in classes}

    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                        title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                        title='Normalized confusion matrix')

    plt.show()
