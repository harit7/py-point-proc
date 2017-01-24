from __future__ import division
import numpy as np
np.random.seed(0)
import utils
from math import log,exp
from basic_models import IntensityBasedPointProcessModel
from collections import defaultdict
import scipy.optimize as optimize

class UnivariateHawkesProcessModel (IntensityBasedPointProcessModel):
    
    def __init__(self,l_0=0.0,alpha=0.0,beta=0.0,all_realizations=defaultdict(list),cur_realization_key=""):
        self.l_0   = l_0
        self.alpha = alpha
        self.beta  = beta
        self.all_realizations = all_realizations
        self.cur_realization_key = cur_realization_key
        

    def print_params(self):
        
        print "l_0 = "+ str(self.l_0)
        print "alpha = "+ str(self.alpha)
        print "beta = "+ str(self.beta)
        
    def lamda(self,t):

        T = self.get_cur_realization()
        s = 0
        for t_j in T:
            if(t_j<t):
                s += exp(-self.beta*(t-t_j))
                
        l = self.l_0 + self.alpha*s

        return l

    def Lamda(self,t):
        
        T = self.get_cur_realization()
        
        s = 0
        for t_j in T:
            if(t_j<t):
                s =  s + 1 - exp(-self.beta*(t-t_j))

        return t*self.l_0 + (self.alpha/self.beta)*s
    
    def __cumm_excitation(self,t):
        
        T = self.get_cur_realization()
        s = 0
        for t_j in T:
            if(t_j<t):
                s += exp(-self.beta*(t-t_j))
        return s
              
    def sample(self,t_max):
        
        if(self.beta<self.alpha):
            raise("Unstable. You must have alpha < beta")
            
        lambda_star = self.l_0
        dlambda     = 0.0
        t=0
        
        #first event
        U  = np.random.uniform(0,1)
        s  = -(1.0 / lambda_star) * log(U)
        T =[]
        if (s <= t_max):
            T.append(s)
            dlambda = self.alpha
            t       = s
        else:
            return T
    
        while (True):
          
            lambda_star = self.l_0 + dlambda*exp(-self.beta*(s-t))
            U  = np.random.uniform(0,1)
            s  = s - (1.0 / lambda_star) * log(U)
           
            if (s > t_max):
                return T
            
            D  = np.random.uniform(0,1)
            if (D <= (self.l_0+dlambda*exp(-self.beta*(s-t))) / lambda_star):
                T.append(s)
                dlambda = dlambda*exp(-self.beta*(s-t)) + self.alpha
                t       = s 
        return T
        
    def __gradient_ll(self,x):
        #[l_0, alpha, beta]
        T = self.get_cur_realization()
        self.l_0 = x[0]
        self.alpha = x[1]
        self.beta = x[2]
        
        g = np.zeros(3)
        lmd_t_i_inv = map(lambda t_i: 1/self.lamda(t_i), T )
        cumm_exc_t_i = map(lambda t_i: self.__cumm_excitation(t_i), T )
        
        cumm_exc_T  = cumm_exc_t_i[-1]
        g[1] =  sum([a*b for a,b in zip(lmd_t_i_inv,cumm_exc_t_i)]) - (1/self.beta)*cumm_exc_T
        
        g[0] = sum(lmd_t_i_inv) + T[0] - T[-1]
        
        cc = []
        for t_i in T:
            s = 0
            for t_j in T:
                if(t_j<t_i):
                    s += (exp(-self.beta*(t_i-t_j))/(t_i-t_j))

            cc.append(0-self.alpha*s)                      
        
        x = self.alpha/(self.beta**2)
        g[2] =  sum([a*b for a,b in zip(lmd_t_i_inv,cc)]) + x + (1/b)*cc[-1] - x * cumm_exc_T
                                         
        return g
        
        
    def fit_mle_params(self):
        
        def obj(x):
            if(min(x)<0):
                return 10000000
            self.l_0   = x[0]
            self.alpha = x[1]
            self.beta  = x[2]
            l = -self.log_likelihood()
            return l
                

        bnds = ((0.0001, 10.0), (0.0001,10.0), (0.0001,10.0))
        res = optimize.minimize(obj, x0= (0.0001,0.0001,0.0001),method="L-BFGS-B",bounds=bnds,tol=1e-18)
        
        self.l_0   = res.x[0]
        self.alpha = res.x[1]
        self.beta  = res.x[2]
        
        return res
        
        
    def fit_map_params(self):
    
        from pymc import Beta, Gamma,MAP
        import pymc
        a = 1.0
        b = 1.0
        alpha = Gamma( 'alpha', alpha=a, beta=b)
        beta  = Gamma( 'beta', alpha=a, beta=b)
        l_0   = Gamma( 'l_0', alpha=a, beta=b)
        
        T = self.get_cur_realization()
        
        @pymc.stochastic(observed=True)
        def custom_stochastic(value=T, l_0= l_0,alpha=alpha, beta = beta ):
            if(l_0<0 or alpha <0 or beta <0):
                return -1000000.0
            self.l_0 = l_0
            self.alpha = alpha
            self.beta = beta
                    
            return self.log_likelihood()


        model = MAP([custom_stochastic,l_0,alpha,beta])
        model.fit()
        c = self.alpha + self.beta
        
        return np.array([self.l_0,self.alpha,self.beta])

   