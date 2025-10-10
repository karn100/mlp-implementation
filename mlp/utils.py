import numpy as np

def one_hot(y,num_classes):
    return np.eye(num_classes)[y]

def accuracy(y_true,y_pred):
    return np.mean(y_true == y_pred)

def dropout(H,p = 0.5):
    mask = (np.random.randn(*H.shape) > p)/(1 - p)
    return H * mask
