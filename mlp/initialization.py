import numpy as np

def xavior_init(n_in:int,n_out:int)-> np.darray: #for Sigmoid and Tanh
    return np.random.rand(n_in,n_out) * np.sqrt(2 / (n_in + n_out))

def he_init(n_in:int,n_out:int)-> np.darray:     #for ReLU and Leaky ReLU
    return np.random.randn(n_in,n_out) * np.sqrt(2/n_in)
