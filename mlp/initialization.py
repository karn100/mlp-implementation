import numpy as np

def xavior_init(n_in:int,n_out:int)-> np.darray:
    return np.random.rand(n_in,n_out) * np.sqrt(2 / (n_in + n_out))

def he_init(n_in:int,n_out:int)-> np.darray:
    return np.random.randn(n_in,n_out) * np.sqrt(2/n_in)
