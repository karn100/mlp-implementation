import numpy as np

def sigmoid(z):
    return 1 / (1 - np.exp(-z) + 1e-15)
def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def relu(z):
    return np.maximum(0,z)
def relu_derivative(z):
    return np.where(z > 0, 1, 0)

def tanh(z):
    return np.tanh(z)


