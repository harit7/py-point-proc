from __future__ import division
import numpy as np

import utils
from basic_models import IntensityBasedPointProcessModel

np.random.seed(0)
import scipy.optimize as optimize
from collections import defaultdict

from math import exp,log

class WeibullHazardProcessModel (IntensityBasedPointProcessModel):
    
    #webull distribution f(x,eta,l_0)= eta*(l_0)^eta *x(eta-1)*exp(-l_0*x^(eta))
    def __init__(self,eta=0.0,l_0=0.0,all_realizations=defaultdict(list),cur_realization_key=""):
        self.l_0 = l_0
        self.eta = eta
        self.all_realizations = all_realizations
        self.cur_realization_key = cur_realization_key

        
    def lamda(self,t):
        
        T     = self.get_cur_realization()
        n_t   = self.N_t(t)
    
        if(n_t<1):
            return 0.0 

        x = self.eta * self.l_0**(self.eta) * utils.safe_pow(t-T[n_t-1],self.eta-1)
        return x
    
    
    def Lamda(self,t):

        T     = self.get_cur_realization()
        n_t   = self.N_t(t)
        
        if( len(T)==0 or n_t ==0):
            return 0.0

        s = 0
        for j in range(0,n_t-1):
            s+= utils.safe_pow( (T[j+1] - T[j]),self.eta)

        ret = (self.l_0**self.eta)*(s + utils.safe_pow(t - T[n_t-1],self.eta))
        return ret
    
    def sample(self,t_max=1000):
        T = []
        
        p = 1.0/self.eta
        y = 1.0/self.l_0
        t =0
        while(t<t_max):
            u = utils.get_unif_random()
            x = y*((-log(u))**(p))
            t = t + x
            T.append(t)
        return T
    
    def fit_mle_params(self):
        
        def obj(x):
            if(min(x)<0):
                return 10000000
            self.l_0   = x[0]
            self.eta   = x[1]
            l = -self.log_likelihood()
            return l
                

        bnds = ((0.0001, 100.0), (0.0001,100.0))
        res = optimize.minimize(obj, x0= (0.0001,0.0001),method="L-BFGS-B",bounds=bnds,tol=1e-18)
        
        self.l_0   = res.x[0]
        self.eta   = res.x[1]
        
        return res
        
        
    def fit_map_params(self):
    
        from pymc import Beta, Gamma,MAP
        import pymc
        a = 1.0
        b = 1.0
        eta = Gamma( 'eta', alpha=a, beta=b)
        l_0 = Gamma( 'l_0', alpha=a, beta=b)
        
        T = self.get_cur_realization()
        
        @pymc.stochastic(observed=True)
        def custom_stochastic(value=T, l_0= l_0,eta=eta ):
            if(l_0<0 or eta <0 ):
                return -1000000.0
            self.l_0  = l_0
            self.eta = eta
                    
            return self.log_likelihood()


        model = MAP([custom_stochastic,l_0,eta])
        model.fit()
        
        return np.array([self.l_0,self.eta])
    
    def print_params(self):
        print "l_0 = "+str(self.l_0)+"\t eta= " +str(self.eta)
    