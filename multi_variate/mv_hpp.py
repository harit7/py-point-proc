from collections import defaultdict
from math import log
import numpy as np

class MultiVariateHomogeneousPoissonProcessModel:
    def __init__(self,dim, l_0=None):
        self.dim = dim
        self.l_0 = l_0
        
    def lamda(self,t):
        return self.l_0
    
    def lamda_c(self,t,c):
        return self.l_0[c]
    
    def neg_log_likelihood(self):
        ll = 0
        C = len(self.sequences)
        for seq in self.sequences:
            for i in seq.dims:
                ll += log(self.l_0[i])* len(seq.TT[i]) - self.l_0[i]* seq.T
        return 0-ll
    
    def fit_mle_params(self,sequences):
        T_sum = sum(map(lambda seq:seq.T, sequences))
        d = defaultdict(int)
        self.sequences = sequences
        for seq in sequences:
            for i in seq.dims:
                d[i]+= len(seq.TT[i])
        self.l_0 = np.zeros(self.dim)
        
        for i in range(self.dim):
            self.l_0[i] = d[i]/T_sum
            if(self.l_0[i]<1e-18):
                self.l_0[i] = 1e-18
                
    def print_params(self):
        print self.l_0
  