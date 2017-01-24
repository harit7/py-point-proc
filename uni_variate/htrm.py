from math import exp,log
import numpy as np
np.random.seed(0)
from basic_models import IntensityBasedPointProcessModel

class HTRm (IntensityBasedPointProcessModel) :
    
    def __init__(self,r_0=0.0,a_s=0.0,b_s=0.0,T_s=[],a_h=0.0,b_h=0.0,kappa=0.0,
                 all_realizations=defaultdict(list),cur_realization_key=""):

        self.r_0 = r_0
        self.a_s = a_s
        self.b_s = b_s
        self.a_h = a_h
        self.kappa = kappa
        
        self.all_realizations = all_realizations
        self.cur_realization_key = cur_realization_key
        
    def lambda_sale(self,t):
        T   = self.get_cur_realization()
        T_s = self.T_s
        
        s   = 0
        for j in range(0,len(T_s)):
            if(T_s[j]<=t):
                s += (t-T_s[j])*exp(- self.b_s*(t-T_s[j]))

        return self.r_0 * (1 + self.a_s * s)

    def Lambda_sale(self,t):
        T   = self.get_cur_realization()
        T_s = self.T_s
        a_s = self.a_s
        b_s = self._b_s
        r_0 = self.r_0

        s = 0
        d = 1.0/self.b_s

        for i in range(0,len(T_s)):
            if(T_s[i]<=t):
                h = max(0, t_0 - T_s[i])
                g = max(0, t   - T_s[i])
                s = s + ( exp(-b_s*h)*(h + d) ) - ( exp(-b_s*g)*( t - T_s[i] + d)  )
        s = r_0*(t-t_0) + (r_0*a_s*d) * s

        return s


    def lambda_excite(self,t,L_t, L_T):

        a_h = self.a_h
        b_h = self.b_h
        T   = self.get_cur_realization()

        s   = 0
        for j in range(0,len(T)):
            if(T[j]<t):
                try:
                    s += exp(-b_h*(L_t-L_T[j]))
                except:
                    print b_h, L_t, L_T[j]

        return (1 + a_h * s)
        

    def Omega(self,t,L_t, L_T):
        
        a_h = self.a_h
        b_h = self.b_h
        T   = self.get_cur_realization()
        
        s = 0
        for j in range(0,len(T)):
            if(T[j]<t):
                try:
                    s =  s + 1 - exp(-b_h*(L_t- L_T[j]))
                except:
                    print "error:" + str((L_t, L_T[j], b_h, T,params, L_T ))

        return L_t+ (float(a_h)/b_h)*s

    
    def lambda_pre(self,t,O_t,O_T):
        T     =  self.get_cur_realization()
        kappa =  self.kappa
        n_t = N_t(t,T)

        if(len(T) ==0 or n_t ==0):

            return 1.0
        if(n_t == 1 and t == T[0]):
            print "n_t 1"
            return kappa*(1.0)
        d = O_t-O_T[n_t-1]
        if(d<0):
            print "delta "+ str(d)
        return kappa * safe_pow(O_t-O_T[n_t-1],kappa-1) 


    def Psi(self,t,O_t,O_T):
        
        T     =  self.get_cur_realization()
        kappa =  self.kappa

        n_t = N_t(t,T)
        if(len(T)==0 or n_t ==0):
            return 0.0

        s = 0    
        for j in range(1,n_t-1):
            #print j,n_t,len(T),len(O_T)
            s += safe_pow(O_T[j+1] - O_T[j],kappa)

        ret = s + safe_pow(O_t - O_T[n_t-1],kappa)

        return ret
    

    def lamda(self,t):


        T = self.get_c
        L_T = map(self.lamda_sale,T)
        O_T = []
        for i in range(len(T)):
            O_T.append(self.Omega(t[i],L_t[i], L_T))
        
        L_t = self.Lambda_sale(t)
        O_t = self.Omega(t, L_t, L_T)
        
        return self.lambda_sale(t)*self.lambda_excite(t,L_t,L_T)*self.lambda_pre(t,O_t,O_T)
    

    def Lamda(self):
        raise("Not Implemented")
        
    def log_lambda_all(self,t,L_T,O_T):
        
        T     =  self.get_cur_realization()
        
        L_t = self.Lambda_sale(t)

        O_t = self.Omega(t,L_t, L_T)

        l_sal_t = self.lambda_sale(t)
        l_exc_t = self.lambda_excite_htr(t,L_t,L_T)
        l_pre_t = self.lambda_pre_htr(t,O_t,O_T)


        if(l_sal_t <0 or l_exc_t<0 or l_pre_t <0):
            print "-ve" + str((l_sal_t,l_exc_t, l_pre_t))

        if(l_sal_t == 0 or l_exc_t==0 or l_pre_t == 0):
            print "zero" + str((l_sal_t,l_exc_t, l_pre_t))

        if(l_sal_t ==0):
            print "l_sal_t=0"
            l_sal_t = 1.0 
        if(l_exc_t ==0):
            print "l_exc_t=0"
            l_exc_t = 1.0
        if(l_pre_t ==0):
            print "l_pre_t=0"
            l_pre_t = 1.0


        return log(l_sal_t) + log(l_exc_t ) + log(l_pre_t)
    
    def log_likelihood(self):
        T     =  self.get_cur_realization()
        L_T = []
        O_T = []
        t_max = T[-1]
        for t in T:
            L_T.append(self.Lambda_sale(t))

        for t in T:
            L_t = self.Lambda_sale(t)
            O_T.append(self.Omega(t,L_t,L_T))

        L_t_mx = self.Lambda_sale(t_max)
        O_t_mx = self.Omega(t_max, L_t_mx, L_T) 

        ll = 0
        for t in T:
            l = self.log_lambda_all(t, L_T, O_T)
            ll += l

        return ll - self.Psi(t_max, O_t_mx, O_T)

    def log_lik_2(T,t_max,params):

        a_h = self.a_h
        b_h = self.b_h

        kappa = self.kappa
        T     =  self.get_cur_realization()

        L_T = map(self.Lambda_sale, T)
        L_t_max = self.Lambda_sale(t_max)

        sale_s = 0
        for t in T:
            sale_s += log(self.lambda_sale(t))

        n = len(T)
        A = np.zeros(n)
        hawk_s = 0
        for j in range(0,n-1):

            A[j+1] = (1+A[j])*exp(-b_h*(L_T[j+1] - L_T[j]))
            hawk_s += log(1+ a_h*A[j])

        hawk_s += log(1+a_h*A[n-1])

        hawk_s_1= 0
        for j in range(0,n):
            L_t       =   self.Lambda_sale(T[j])
            hawk_s_1 += log(self.lambda_excite(T[j],L_t,L_T))

        pre_s = 0
        D_omg = np.zeros(n)
        s1 = 0
        s2 = 0


        for j in range(1,n-1):
            D_omg[j] = L_T[j+1] - L_T[j] + (a_h/b_h)*(1 + A[j] - A[j+1])
            '''
            if(L_T[j+1]-L_T[j]<0):
                print L_T[j+1],L_T[j]
            if(1 + A[j] - A[j+1]<0):
                print  A[j],A[j+1]
            '''    
            if(D_omg[j]<0):
                print"D_omg"+ str((D_omg[j],L_T[j+1],L_T[j],A[j],A[j+1]))
                D_omg[j]=1.0
            if(D_omg[j]==0):
                D_omg[j]=1.0
            s1 += log(D_omg[j])
            s2 += (D_omg[j])**(kappa)

        pre_s = (n-1)*log(kappa) + (kappa-1)* s1 - s2
        # O_T[1] = L_T[1]  eqn 18

        ex = L_t_max - L_T[-1] + (a_h/b_h)*(1+ A[-1])*(1- exp(-b_h*(L_t_max- L_T[-1]))) 

        #print sale_s, hawk_s, pre_s, L_T[1]**kappa, ex**kappa
        #print hawk_s_1
        return sale_s + hawk_s + pre_s - L_T[1]**kappa - ex**kappa
    
    
    def func(t):
        global d_lam_j
        global lam_j_1

        lam = self.Lambda_sale(t)

        return lam - lam_j_1 - d_lam_j   


    def gradient(t):
        r_0  = self.r_0
        a_s  = self.a_s
        b_s  = self.b_s

        s   = 0
        for j in range(0,len(T_s)):
            if(T_s[j]<=t):
                s += (t-T_s[j])*exp(-b_s*(t-T_s[j]))

        return r_0 * (1 + a_s * s)


    def sample(t_max=1000):
        T    =  []
        T_s  = self.T_s
        
        global d_lam_j
        global lam_j_1

        self.__d_lam_j = 0.0

        psi     = [0]
        omg     = [0]

        b_h     = self.b_h
        a_h     = self.a_h
        kappa   = self.kappa

        #T       = np.zeros(n_samples)
        T[0]    = self.t_0 

        d_psi = np.random.exponential(1.0, n_samples+1)

        d_psi[0]= 0

        d_omg = d_psi**(1.0/kappa)

        d_lam = np.zeros(len(d_omg))

        d_lam[0] = 0.0

        B = a_h/b_h

        for j in range(1,n_samples):

            self.__lam_j_1  = Lambda_sale(T[j-1],T,params)

            d_lam[j]     = d_omg[j] - B + (1.0/b_h) *lambertw( b_h * B * exp(-b_h * ( d_omg[j]-B) ) ).real 
            B = (a_h/b_h) * ( 1 + B * exp(-b_h * d_lam[j]) )
            d_lam_j = d_lam[j]

            t = bisect(func,T[j-1],T[j-1]+1000)

            #t = newton(func,T[j-1], tol=0.001,maxiter=100000)
            T[j] = t
            #T.sort() # to find the t_j in the neighbourhood of t_{j-1} 
        return T

    def next_sample(k):
        global p
        global T
        global T_s
        global d_lam_j
        global lam_j_1

        d_lam_j = 0.0

        p       = params.copy()

        T_s     = p["T_s"]


        psi     = [0]
        omg     = [0]

        b_h     = p["b_h"]
        a_h     = p["a_h"]
        kappa   = p["kappa"]


        n       = len(cur_samples)
        T       = np.zeros(n+k)
        np.copyto(T[:n],cur_samples)

        d_psi   = np.random.exponential(1.0, k+len(T)+1)
        d_psi[0]= 0

        d_omg   = d_psi**(1.0/kappa)

        d_lam   = np.zeros(len(d_omg))

        d_lam[0]= 0.0

        B       = a_h/b_h

        for j in range(1,n+k):

            lam_j_1  = Lambda_sale(T[j-1],T[:j],params)

            d_lam[j] = d_omg[j] - B + (1.0/b_h) *lambertw( b_h * B * exp(-b_h * ( d_omg[j]-B) ) ).real 
            B        = (a_h/b_h) * ( 1 + B * exp(-b_h * d_lam[j]) )
            d_lam_j  = d_lam[j]

            if( j>= len(cur_samples) ):

                t    = newton(func,T[j-1]+1, tol=0.001,maxiter=100000)
                #t =  bisect(func,T[j-1],T[j-1]+100)

                T[j] = t

        return T[n:]


