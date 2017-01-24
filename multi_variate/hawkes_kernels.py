from math import log,exp,sqrt,pi
from scipy.special import erf
from math import pi

root_2pi = sqrt(2*pi)
root_2   = sqrt(2)



def gaussian_pdf(x,mu,sigma):
    return (1.0/(sigma*root_2pi))*(exp( - ((x-mu)/root_2*sigma)**2) )
def gaussian_cdf(x,mu,sigma):
    return (1.0/2)*(1 + erf((x-mu)/(sigma*self.__root_2)))


class MixtureGaussianKernel:
    def __init__(self, k,w , mu, sigma):
        '''
          mixture of k gaussians. |mu| = |sigma| =k
        '''
        self.k = k
        self.mu = mu
        self.w = w
        self.sigma = sigma
        self.__root_2pi = sqrt(2*pi)
        self.__root_2 = sqrt(2)
    def kernel(self,x):
        '''
            g(x) = \sum \limits_{j=1}^k \frac{1}{\sigma \root{2\pi}} \exp( \frac{(x-\mu[j])^2}{2\sigma^2})
        '''
        def gaussian_pdf(x,mu,sigma):
            return (1.0/(sigma*self.__root_2pi))*(exp( - (x-mu)**2/2*sigma**2))
        
        return sum(map(lambda i: w[i]*gaussian_pdf(x,self.mu[i],self.sigma[i]), range(self.k)))
    
    def kernel_integral(self,x):
        '''
            G(x) = \int \limits_0^x g(t) dt
        '''
        def gausian_cdf(x,mu,sigma):
            return (1.0/2)*(1 + erf((x-mu)/(sigma*self.__root_2)))
        
        return sum(map(lambda i: w[i]*gausian_cdf(x,self.mu[i],self.sigma[i]),range(self.k) ))

    def kernel_grad(self,x):
        return None
    
    def kernel_integral_grad(self,x):
        return None
    
    def get_num_params(self):
        return self.k*3 
    
    def set_params_callback(self,v):
        self.w = v[:k]
        self.mu = v[k:k+k]
        self.sigma = v[2*k:]
        
        
class ExponentialKernel:
   
    def __init__(self,beta):
        assert beta >0
        self.beta = beta

        
    def kernel(self,x):
        try:
            return exp(-self.beta*x)
        except:
            print self.beta,x
            raise
    
    def kernel_integral(self,x):
        bt = self.beta
        return (1/bt)*(1 - exp(-bt*x))
    
    def kernel_grad(self,x):
        return 0-x* exp(-self.beta*x)
    
    def kernel_integral_grad(self,x):
        ebx = exp(-self.beta*x)
        b2_inv = self.beta**2
        return (x/self.beta)*(ebx) + (b2_inv)*(ebx) - b2_inv
        
    
    def get_num_params(self):
        return 1
    
    def set_params_callback(self,vec):
        '''
            vec of length = get_num_params() will be passed by the caller
            it is kernel's responsibility to set the params appropriatly 
        '''
        self.beta =vec[0]
        #self.beta = np.reshape(vec,(self.r,self.c))

    def get_bounds(self):
        return [(1.0,10.0)]
    
    def get_init_params(self):
        return [1.0]
    
    def print_params(self):
        print "beta: "+ str(self.beta)
    

    
class RayleighKernel:
    
    def __init__(self,sigma):
        assert sigma>0
        self.sigma = sigma
            
    def kernel(self,dt):
        try:
            s = self.sigma
            s2 = s**2
            return (dt/s2) *( exp(-dt**2/(2*s2)) )
        except:
            print self.sigma,dt
            raise
    
    def kernel_integral(self,x): # 0 to t
        s = self.sigma**2
        
        return 1 - exp(-x**2/2*s2)
    def kernel_grad(self,x):
        return None
    
    def kernel_integral_grad(self,x):
        return None

    def get_num_params(self):
        return 1
    
    def set_params_callback(self,vec):
        '''
            vec of length = get_num_params() will be passed by the caller
            it is kernel's responsibility to set the params appropriatly 
        '''
        self.sigma = vec[0]
    def get_bounds(self):
        return [(1e-2,100)]
    
    def get_init_params(self):
        return [1.0]
        
    def print_params(self):
        print "sigma: "+ str(self.sigma)

def safe_pow(x,y):
    #x+=1e-10
    #if(x<=0):
        #x = 1e-10
    #    print "pow log: x = "+str(x)
    z = 0
    try:
        z =np.power(x,y)
    except:
        print x,y
        raise
    
        
    return z
    
        
class WeibullKernel:
    '''
    $ \gamma \theta x^{\gamma -1} e^{-\theta x ^{\gamma}}$
    '''
    def __init__(self,gamma,theta):
        assert gamma>0 and gamma <5
        assert theta>0 and theta <5
        self.gamma =gamma # 0 gamma
        self.theta =theta # 1 theta
        
    def kernel(self,x):
        try:
            h = safe_pow(x, (self.gamma-1))
            return self.gamma*self.theta * h * exp( -self.theta * h *x)
        except:
            self.print_params()
            print x
            raise
    
    def kernel_integral(self,x):
        #from 0 to x, not on parameter beta
        # $G(x) = - e^{-\theta x^{\gamma}}$
        return  1 - exp(-self.theta * safe_pow(x, self.gamma))
    
    def kernel_grad(self,x):
        
        #grad w.r.t. parameter beta
        #$$
        #0:gamma, 1: theta
        x += 1e-10
        h       = safe_pow(x, self.gamma)
        e       = exp(-self.theta*h)
        grad    = np.zeros(2)
        grad[0] = self.theta * (h/x) * e * ( 1 +  self.gamma*log(x)*(1-self.theta*h))
        grad[1] = self.gamma*(h/x)*e*(1 - self.theta*h)
        return grad
    
    def kernel_integral_grad(self,x):
        #grad w.r.t. parameter beta
        
        #$ \frac{\partial G}{\partial \gamma}= e^{-\theta x^{\gamma}} \theta (x^\gamma \log (x))$
        #$ \frac{\partial G}{\partial \theta} = e^{-\that x^{\gamma}} x^{\gamma}$
        x += 1e-10
        h       = safe_pow(x, self.gamma)
        e       = exp(-self.theta*h)
        grad    = np.zeros(2)
        grad[0] = self.theta * e * (x**self.gamma*log(x))
        grad[1] = e*h 
        return grad
    
        
    
    def get_num_params(self):
        return 2
    
    def set_params_callback(self,vec):
        '''
            vec of length = get_num_params() will be passed by the caller
            it is kernel's responsibility to set the params appropriatly 
        '''
        self.gamma = vec[0]
        self.theta = vec[1]
        #self.beta = np.reshape(vec,(self.r,self.c))

    def get_bounds(self):
        return [(1e-4,5),(1e-4,5)]
    
    def get_init_params(self):
        
        return [self.gamma,self.theta] #exponentail kernel at (1,1)
    
    def print_params(self):
        print "gamma = "+ str(self.gamma)+"\t theta = "+str(self.theta)        
    

