from collections import defaultdict

class MVPPSequence:
    """
        Class to represent a sequence(realisation) of multivariate point process. 
        All the events in the sequence are in ascending order of time.
        
        Attributes:
        seqId: unique id of the sequence (int)
        lstEvents: list of MVPPEvent objects sorted in ascending order of time.
        T : The horizon of this sequence. default, time of last event. 
        t_0: Time from which the sequence was observed. default, time of first event.
        TT: dictionary with dimensions as keys and the list of events in that dimension as values.
        dims: set of dims of events of this sequence.
        
    """
    def __init__(self,seqId, lstEvents):
        """
        lstEvents must be sorted in ascending order of time of events.
        """
        assert all(lstEvents[i].time <= lstEvents[i+1].time for i in xrange(len(lstEvents)-1))
       
        self.lstEvents = lstEvents
        self.seqId = seqId
        self.T = 0
        self.t_0 = 0
        self.TT = self.detangle()
        self.dims = self.TT.keys()
        
        if(len(lstEvents)>0):
            self.T = lstEvents[-1].time
            self.t_0 = lstEvents[0].time
        
        
    def __str__(self):
        s = ""
        for e in self.lstEvents:
            s += str(e) +","
        return s.rstrip(",")
    
    def detangle(self):
        
        TT = defaultdict(list)   
        for e in self.lstEvents:
            TT[e.dim].append(e.time)
            #self.T = max(self.T,e.time)
            #self.t_0 = min(self.t_0, e.time)
   
        return TT

    def count_dim(self):
        return len(self.TT.keys())
    
    def append(self,evt):
        self.lstEvents.append(evt)
        self.TT[evt.dim].append(evt.time)
        self.T = max(self.T,evt.time)
        self.dims = self.TT.keys()