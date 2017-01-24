from collections import defaultdict

from math import log,exp,sqrt
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import types
import numpy as np
from hawkes_kernels import ExponentialKernel
from mvpp_sequence import MVPPSequence
from mvpp_event import MVPPEvent

class MultiVariateHawkesProcessModel:
    
    def __init__(self,dim,l_0=None,A=None,trig_kernel=None):

        assert dim>0
        self.dim = dim
        self.l_0 = l_0
        self.A   = A
        self.trig_kernel = trig_kernel
        
        if(self.l_0 is None):
            self.l_0 = np.zeros(self.dim)
        if(self.A is None):
            self.A = np.zeros(shape=(self.dim,self.dim))
        
        if(trig_kernel is None):
            self.trig_kernel = [[ExponentialKernel(beta=(1.0/72))
                                 for x in range(self.dim)] for y in range(self.dim)] 
        self.num_kernel_params = 0
        for i in range(self.dim):
            for j in range(self.dim):
                self.num_kernel_params +=  self.trig_kernel[i][j].get_num_params()
        self.eps = 1e-5
        self.fit_kernel_params = False
        self.num_params = self.dim + self.dim*self.dim #+ self.num_kernel_params
        
        self.sequences = None
            
    
    # c is the index of dimension            
    def lamda_c(self,t,c,seq):
        
        #s = self.l_0[c] + self.eps #log safe
        sum_d = np.zeros(self.dim) # contrib of each dimension till t
        
        for e in seq.lstEvents:
            if(e.time < t):
                sum_d[e.dim] += self.trig_kernel[e.dim][c].kernel(t-e.time)
            
        return self.l_0[c] + self.eps + np.dot(sum_d,self.A[:,c]), sum_d
    
         
    #returns a d dim vector           
    def lamda(self,t,seq):
        l = np.zeros(self.dim) + self.l_0
        for e in seq.lstEvents:
            if(e.time<t):
                for d in xrange(self.dim):
                    l[d] += self.A[e.dim][d] * (self.trig_kernel[e.dim][d].kernel(t - e.time))
            else:
                break
        return sum(l), l
                
    
    def lamda_ub(self,t,l_t, seq):
        '''
        v = np.arrange(t,t+l_t,1)
        lt = map(lambda x: self.lamda(x,seq)[0], v)
        return max(lt)
        '''
        t1 = time.time()
        l = np.zeros(self.dim) + self.l_0
        for e in seq.lstEvents:
            if(e.time<t):
                for d in xrange(self.dim):
                    l[d] += self.A[e.dim][d] * (self.trig_kernel[e.dim][d].kernel(t - e.time))
            else:
                break
        t2 = time.time()
        return sum(l)
    
    #def Lambda(self,t,seq):
        
    # Should be used to compute log likelihood of any generic kernel
    def neg_log_likelihood_general(self,sequences):
        C = len(sequences)
        ll = 0
        
        grad = None
        if(self.fit_kernel_params):
            grad = np.zeros(self.num_params + self.num_kernel_params)
        else:
            grad = np.zeros(self.num_params)
            
        p    = 0
        grad_kernel = None
        grad_l0 = None
        grad_A = None
        tmp_grad_kernel = None
        if(self.fit_kernel_params):
            p = self.num_kernel_params
            grad_kernel = grad[0:p]
            grad_l0 = grad[p:p+self.dim]
            grad_A = np.reshape(grad[self.dim: (self.dim+1)*self.dim],(self.dim,self.dim))

            tmp_grad_kernel = [[np.zeros(self.trig_kernel[i][j].get_num_params())
                                for j in range(self.dim)] for i in range(self.dim)] 
        else:
            grad_l0 = grad[p:p+self.dim]
            grad_A = np.reshape(grad[self.dim: (self.dim+1)*self.dim],(self.dim,self.dim))
        
        
        for seq in sequences:
            res = 0
            TT  = seq.TT
            m_T = seq.T
            dims = TT.keys()

            for m in dims:
                if(len(TT[m])==0): continue

                for t in TT[m]:
                    lmda_d_t,sum_d = self.lamda_c(t,m,seq)
                    res+=log(lmda_d_t)
                    grad_l0[m]+= 1.0/lmda_d_t
                    grad_A[:,m]+= sum_d/lmda_d_t
                    
                    if(self.fit_kernel_params):
                        for e in seq.lstEvents:
                            if(e.time < t):
                                 tmp_grad_kernel[e.dim][m]+= self.A[e.dim][m]*self.trig_kernel[e.dim][m].kernel_grad(t-e.time)

                        for e in seq.lstEvents:
                            if(e.time<t):
                                tmp_grad_kernel[e.dim][m] /=lmda_d_t
                    
                _sum =0       
                for n in dims:
                    for k in range(len(TT[n])):
                        G = self.trig_kernel[n][m].kernel_integral(m_T - TT[n][k])
                        _sum = _sum + (self.A[n][m])*G
                        
                        G_g = self.trig_kernel[n][m].kernel_integral_grad(m_T - TT[n][k])
                        grad_A[n][m] -= G
                        
                        if(self.fit_kernel_params):
                            tmp_grad_kernel[n][m]-=G_g

                res = res - self.l_0[m] * m_T - _sum
                grad_l0[m] -= m_T
            ll += res
        if(self.fit_kernel_params):
            p =0
            for i in range(self.dim):
                for j in range(self.dim):
                    r = self.trig_kernel[i][j].get_num_params()
                    grad_kernel[p:p+r] = tmp_grad_kernel[i][j]
                    p+=r
        return 0-ll, 0-grad
    
    def grad(self,sequences):
        return self.neg_log_likelihood_general(sequences)[1]
    
    def opt_callback_set_params(self,x):
        p = 0
        if(self.fit_kernel_params):
            for i in range(self.dim):
                for j in range(self.dim):
                    c = self.trig_kernel[i][j].get_num_params()
                    self.trig_kernel[i][j].set_params_callback(x[p:p+c])
                    p+=c

        self.l_0  = x[p:p+self.dim]
        p        += self.dim

        self.A = np.reshape(x[p:],(self.dim,self.dim))
    
    def opt_callback_get_init_x(self):
        n = 0
        x0 = []
        if(self.fit_kernel_params):
            for i in range(self.dim):
                for j in range(self.dim):
                    x0.extend(self.trig_kernel[i][j].get_init_params())

        n = self.dim + self.dim*self.dim        
        x0.extend((1e-8)*np.ones(n))

        return np.array(x0)
    
    def opt_callback_get_bounds(self):
        bnds = []
        if(self.fit_kernel_params):
            for i in range(self.dim):
                for j in range(self.dim):
                    bnds.extend(self.trig_kernel[i][j].get_bounds())
        n = self.dim + self.dim*self.dim        
        bnds.extend([(1e-8,10.0)]*(n))
        return bnds
        
    def print_params(self):
        if(self.fit_kernel_params):
            for i in range(self.dim):
                for j in range(self.dim):
                    self.trig_kernel[i][j].print_params()
        print "l_0: " +str(self.l_0)
        print "A:" + str(self.A)
    
    def opt_callback_obj_grad(self,sequences):
        nll,g = self.neg_log_likelihood_general(sequences)
        return nll,g
    '''
    def fit_mle_params(self,sequences,max_iters=1000,fit_kernel_params=False,tol=1e-18):
        self.sequences = sequences
        C = len(self.sequences)
        self.fit_kernel_params = fit_kernel_params
        self.f = 0
        def obj(x):
            self.opt_callback_set_params(x)
            ll,g = self.neg_log_likelihood_general(sequences)
            #print ll
            self.f+=1 
            print ll/C, self.f
            return ll/C,g/C
        
        def jac(x):
            self.opt_callback_set_params(x)
            g = self.grad()
            #print x.shape, g.shape
            return g/C
        
        bnds = self.opt_callback_get_bounds()
        x0   = self.opt_callback_get_init_x()
        
        res = optimize.minimize(obj, x0= x0,jac=True,method="L-BFGS-B",bounds=bnds,
                                tol=tol,options={"disp":True,"maxiter":max_iters,"maxfun":max_iters})
        
        self.opt_callback_set_params(res.x)
    '''        
        
    '''
    def fit_mle_params(self,prefix=None,df_pd_train=None,dc=None,tol=1e-18,fit_kernel_params=True):
        mle_obj = mle.MaximumLikelihoodEstimator(model = self,prefix=prefix, df_pd_train=df_pd_train,
        dc=dc,tol=tol)
        self.fit_kernel_params = fit_kernel_params
        mle_obj.estimate()
    '''     
    def predictNextEventTime(self,seq,num_sim,step=1):
        ot = OgataThinning()
        t = 0
        for i in range(num_sim):
            event = ot.SimulateNext(self,1000,seq,step)
            t += event.time
        return float(t) / num_sim
    def predictNextKEvents(self,seq,k=5,num_sim=25,step=1):
        n = len(seq.lstEvents)
        import copy
        seq = MVPPSequence(copy.deepcopy(seq.lstEvents))
        
        for j in range(k):
            t = self.predictNextEventTime(seq,num_sim,step)
            s,l= self.lamda(t,seq)
            i = np.argmax(l)
            e = MVPPEvent(time=t,dim=i)
            print j,e
            seq.append(e)
        return seq.lstEvents[n:]
    
    #def predictItem(self,seq, t, num_sim):
        
    
    def plot(self,seq):
        
        def __plot(T,d,t_min,t_max,label):
            print label
            fig = plt.figure(figsize=(20,5))
            ax1 = fig.add_subplot(311)

            ax1.set_xlim([t_min, t_max])

            v = np.arange(t_min,t_max, 0.1)
            v = v[v<=t_max]
            lamdas = map(lambda t: self.lamda_c(t,d,seq)[0], v)

            a1, = ax1.plot(v,lamdas,label=label)

            ax3 = fig.add_subplot(312)
            ax3.set_xlim([t_min, t_max])
            for t_i in T:
                ax3.axvline(t_i)
            plt.legend(handles=[a1])
            plt.show()
            
        TT = seq.TT
        dim_id_name = dict()
        for e in seq.lstEvents:
            dim_id_name[e.dim] = e.dim_name
        t_0 = seq.t_0-100
        T   = seq.T+100
                         
        for d in range(self.dim):
            if(len(TT[d])>0):
                label = dim_id_name[d]
                __plot(TT[d],d,t_0,T,label)
    '''
    def check_stability(self):
        if(self.trig_kernel== self.exp)            
    '''       
