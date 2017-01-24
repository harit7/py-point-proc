from __future__ import division

from math import log,exp
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import scipy.stats as stats
import statsmodels.api as sm
import utils

class IntensityBasedPointProcessModel(object):
    
    def __init__(self,all_realizations=defaultdict(list),cur_realization_key=""):
        self.all_realizations = all_realizations
        self.cur_realization_key = cur_realization_key
        
    def lamda(self,t):
        raise NotImplementedError( "Needs to be implemented in Specific Point Process" )
        
    def Lamda(self,t):
        raise NotImplementedError( "Needs to be implemented in Specific Point Process" )
        
    def add_realization(self,realization_key="", realization_data=[]):     
        self.all_realizations[realization_key] = realization_data
        
    def get_cur_realization(self):
        return self.all_realizations[self.cur_realization_key]
    
    def set_cur_realization_key(self,key=""):
        self.cur_realization_key = key
        
    def N_t(self,t):
        T   = self.get_cur_realization()
        n_t = 0
        for j in range(0,len(T)):
            if(T[j]<t):
                n_t+=1
        return n_t
    
    def fit_mle_params(self):
        raise NotImplementedError( "Needs to be implemented in Specific Point Process" )
        
    def log_likelihood(self,realization_key=None):
       
        T = []
        if(realization_key==None):
            T = self.get_cur_realization()
        else:
            T = self.all_realizations[realization_key]
            
        s = sum(map(lambda t: utils.safe_log(self.lamda(t)),T))
        return s - self.Lamda(T[-1]) + self.Lamda(T[0])
    
    def sample_1(self,n_samples=100,t_min=0,t_max=100,prec=0.1):
        print n_samples
        x = np.random.uniform(0,1,n_samples)
        print x
        s = 0
        v = np.arange(t_min,t_max,prec)
        T = []

        e = 0
        f = 0

        for u in x:
            s     = s - log(u)
            x     = np.array(map(self.Lamda, v))

            try:
                t = min(v[np.where(x>=s)[0]])
                T.append(t)
            except:
                e+=1
        print e
        T.sort()
        return T

    
    # will only use parameters, not the history given in T
    def sample(self,n_samples=100,t_min=0,t_max=100,prec=0.1):
        l_t = 10 # inf
        n = 0
        t = 0
        T = []
        
        while t<t_max and n < n_samples:
            v = np.arange(t,t+l_t,prec)
            m_t = max(map(self.lamda, v))

            s = np.random.exponential(1/m_t)
            U = np.random.uniform(0,1)
            
            if(s>l_t):
                t = t + l_t
            elif(t+s > t_max or U > self.lamda(t+s)/m_t):
                t = t+s
            else:
                T.append(t+s)
                t = t+s
                n+=1    
        return T
    
    
    def plot(self):
        
        T = self.get_cur_realization()
        
        if(len(T)==0):
            print "No events to plot"
            return
        
        t_max = max(T)
        t_min = min(T)

        fig = plt.figure(figsize=(20,5))
        ax1 = fig.add_subplot(311)
        ax1.set_xlim([t_min, t_max])

        v = np.arange(t_min,t_max, 0.1)
        v = v[v<=t_max]
        lamdas = map(self.lamda, v)

        ax1.plot(v,lamdas)

        ax3 = fig.add_subplot(312)
        ax3.set_xlim([t_min, t_max])
        for t_i in T:
            ax3.axvline(t_i)

        plt.show()
        
    def plot_iei_density(self,):
        raise NotImplementedError("Implement")
    
    def to_iei(self):
        T = self.get_cur_realization()
        return [t_i - T[i - 1] for i, t_i in enumerate(T)][1:]
    
    def coef_variation(self):
        #T = get_cur_realization(self)
        iei = self.to_iei()
        cv  = stats.variation(iei)
        return cv
    
    def iei_corr(self):
        r = {
                "spearmanr":{"coef":0.0,"p-value":0.0},
                "pearsonr": {"coef":0.0,"p-value":0.0}
            }
        iei = self.to_iei()
        r1 = stats.spearmanr(iei[:-1],iei[1:])
        r["spearmanr"]["coef"]=r1.correlation
        r["spearmanr"]["p-value"]=r1.pvalue
        
        r2 = stats.pearsonr(iei[:-1],iei[1:])
        r["pearsonr"]["coef"] = r2[0]
        r["pearsonr"]["p-value"] = r2[1]
        
        return r
    
    def qq_plot(self,ax=None,dist=stats.expon): 

        L = map(self.Lamda, self.get_cur_realization())
        g = [x-L[i-1] for i,x in enumerate(L)][1:]
        sm.qqplot(np.array(g),line='r',ax=ax)
        
             
class HomogeneousPoissonProcessModel (IntensityBasedPointProcessModel):
    
    def __init__(self,l_0=0.0,all_realizations=defaultdict(list),cur_realization_key=""):
        self.l_0 = l_0
        self.all_realizations = all_realizations
        self.cur_realization_key = cur_realization_key

        
    def lamda(self,t):
        return self.l_0
    
    def Lamda(self,t):
        return self.l_0*t
    
    def fit_mle_params(self):
        T = self.get_cur_realization()
        
        if(len(T)>0):
            self.l_0 = len(T)/(T[-1]-T[0])
            
    def print_params(self):
        print "l_0= "+str(self.l_0)
    
    def sample(self,t_max=1000):
        assert self.l_0 > 0
        T = []

        u = utils.get_unif_random(0,1)
        t = -log(u)/self.l_0
        T.append(t)
        
        while(t<t_max):
            u = utils.get_unif_random(0,1)
            t = T[-1] -log(u)/self.l_0
            T.append(t)
            
        return T

class GeneralNonHomogeneousPoissonProcessModel( IntensityBasedPointProcessModel):
    
    def __init__(self,lamda,Lamda,all_realizations=defaultdict(list),cur_realization_key=""):
        self._lamda = lamda
        self._Lamda = Lamda
        self.all_realizations = all_realizations
        self.cur_realization_key = cur_realization_key
        
    def lamda(self,t):
        return self._lamda(t)
    
    def Lamda(self,t):
        return self._Lamda(t)    
        

    