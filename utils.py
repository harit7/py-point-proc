import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
from math import log,exp

def get_unif_random(low=0.0,high=1.0):
    x = 0
    i = 0
    while(x==0):
        x = np.random.uniform(0.0,1.0)
        i+=1
    return x

def is_sorted(l):
    return all(l[i] <= l[i+1] for i in xrange(len(l)-1))

def safe_pow(x,p):

    if(x==0):
        return 1.0 #
    elif(x<0 and p-int(p)>0.0):
        #print "ERR: " +str(x) +"\t"+str(p)
        return 0.0 #????
    else:
        #print "ret :" +str(x**p)
        return x**p
def safe_log(x):
    if(x==0.0):
        return 0.0
    return log(x)

def to_iei(T):
    return [t_i - T[i - 1] for i, t_i in enumerate(T)][1:]

def coef_variation(T):
    #T = get_cur_realization(self)
    iei = to_iei(T)
    cv  = stats.variation(iei)
    return cv

def iei_corr(T):
    r = {
            "spearmanr":{"coef":0.0,"p-value":0.0},
            "pearsonr": {"coef":0.0,"p-value":0.0}
        }
    iei = to_iei(T)
    r1 = stats.spearmanr(iei[:-1],iei[1:])
    r["spearmanr"]["coef"]=r1.correlation
    r["spearmanr"]["p-value"]=r1.pvalue

    r2 = stats.pearsonr(iei[:-1],iei[1:])
    r["pearsonr"]["coef"] = r2[0]
    r["pearsonr"]["p-value"] = r2[1]

    return r

class Result:
   
    def __init__(self,test_ll=0.0,train_ll=0.0,all_ll=0.0,model=None):
        self.test_ll = test_ll
        self.train_ll = train_ll
        self.all_ll = all_ll
        self.model = model
        
    def __str__(self):
        return "test_ll = "+str(self.test_ll) +"\t all_ll = "+str(self.all_ll)+"\t train_ll= "+str(self.train_ll)
    
def test_models(models,T,train_percent=0.8):
    
    l_r = []
    for model in models:
        r  = Result()
        n  = int(len(T)*train_percent)
        model.add_realization("all",T)
        model.add_realization("train",T[:n])
        model.add_realization("test",T[n:])
        
        model.set_cur_realization_key("train")
        model.fit_mle_params()
        model.set_cur_realization_key("all")
        
        r.train_ll = model.log_likelihood("train")
        r.test_ll  = model.log_likelihood("test")
        r.all_ll   = model.log_likelihood("all")
        l_r.append(r)
        
    return l_r
        
        
    