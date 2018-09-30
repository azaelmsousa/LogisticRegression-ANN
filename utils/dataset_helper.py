import numpy as np 


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