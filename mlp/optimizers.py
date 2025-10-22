import numpy as np
import torch
from torch.optim import Optimizer

class MomentumOptimizer(Optimizer):
    def __init__(self,params,lr,mom_cof = 0.9,nesterov = False):

        default = dict(lr = lr,mom_cof = mom_cof,nesterov = nesterov)
        super().__init__(params,default)
    
    def step(self,closure = None):
        loss = None
        if closure:
            loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            mom_cof = group['mom_cof']
            nesterov = group['nesterov']
            for  p in group['params']:
            