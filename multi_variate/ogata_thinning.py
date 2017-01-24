import time
from mvpp_sequence import MVPPSequence
from mvpp_event import MVPPEvent
import numpy as np

class OgataThinning:
           
    def Simulate(self,process,seqId,T_c):
        seq = MVPPSequence(seqId,[])
        t = 0
        l_t = 10
        while(t < T_c):
            m_t      = process.lamda_ub(t, l_t, seq)
            s        = np.random.exponential(1/m_t)
            u        = np.random.uniform(0,1)
            #lamda_t,lamda_dim = 
            if(s > l_t):
                t += l_t
            elif( u > (process.lamda(t + s, seq)[0] / m_t)):
                t += s
            else:
                t += s
                lmda = process.lamda(t,seq)[1]
                d = np.argmax(lmda)
                seq.append(MVPPEvent(time=t,dim=d))
                    
        return seq

    def SimulateNext(self,process, horizon, in_seq,step=1):
        lstEvents = in_seq.lstEvents
        evt = MVPPEvent()
        l_t = step
        if(len(lstEvents) > 0):
            t1 = time.time()
            t_n = lstEvents[-1].time
            t = t_n
            evt.time = t_n + horizon
            while(t< t_n+horizon):
                #print t, t_n, t_n+ horizon
                m_t      = process.lamda_ub(t, l_t, seq)
                s        = np.random.exponential(1/m_t)
                u        = np.random.uniform(0,1)
                #lamda_t,lamda_dim = 
                if(s > l_t):
                    t += l_t
                elif( u > (process.lamda(t + s, seq)[0] / m_t)):
                    t += s
                else:
                    t2 =time.time()
                    t+=s
                    lmda = process.lamda(t,seq)[1]
                    d    = np.argmax(lmda)
                    evt.time = t
                    evt.dim = d
                    t3 =time.time()
                    #print t2-t1, t3-t2
                    return evt
        return evt

    