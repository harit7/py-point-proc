class MVPPEvent:
    """
        Class for Mutlivariate Point process event. An MVPPEvent object e1 is less than another object(e2) if the e1.time < e2.tim
        
        Attributes: 
        
        dim  : dimension in which the event occurred (int)
        time : time at which the event occurred (float)
        dim_name : optional name/label of the dimension. (string)
    """
    
    def __init__(self, dim=0 , time=0, dim_name=None):
        self.dim = dim
        self.time = time
        self.dim_name = dim_name
        
    def __str__(self):
        s = str(self.dim)+"*"+str(self.time)
        if(not self.dim_name is None):
            s+= "*"+self.dim_name
        return s
    
    def __gt__(self,e2):
        return self.time > e2.time
    
    def __eq__(self,e2):
        return self.time == e2.time and self.dim == e2.dim
    
    def __lt__(self,e2):
        return self.time < e2.time
    
    def __le__(self,e2):
        return self.time<=e2.time
    
    def __cmp__(self,e2):
        return self.time <= e2.time
    
    def __hash__(self):
        return hash(str(self.dim)+","+str(self.time))