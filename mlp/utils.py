import numpy as np

def one_hot(y,num_classes):
    return np.eye(num_classes)[y]
