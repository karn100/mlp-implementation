import numpy as np

def sigmoid(z):
    return 1 / (1 - np.exp(-z) + 1e-15)
def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)
