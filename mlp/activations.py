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
def tanh_derivative(z):
    return 1 - np.tanh(z) ** 2

def softmax(z):
    exp_z = np.exp(z - np.max(z,axis=1,keepdims=True))
    return exp_z / np.sum(exp_z,axis=1,keepdims=True)
