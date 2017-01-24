import scipy.optimize as optimize
import numpy as np

class Optimizer:
    
    def __init__(self,model):
        self.model = model
        
    def sequential_fit(self,sequences,max_iters=1000,tol=1e-18):
        C = len(sequences)
        model = self.model
        def __obj(x):
            model.opt_callback_set_params(x)
            obj,grad = model.opt_callback_obj_grad(sequences)
            print obj/C
            return obj/C,grad/C
        

        bnds = model.opt_callback_get_bounds()
        x0   = model.opt_callback_get_init_x()

        res = optimize.minimize(__obj, x0= x0,jac=True,method="L-BFGS-B",bounds=bnds,
                                tol=tol,options={"disp":True,"maxiter":max_iters,"maxfun":max_iters})

        model.opt_callback_set_params(res.x)
        
    def distributed_fit(self,rdd_instances,max_iters=1000,tol=1e-18):
        model          = self.model
        #df_spark = dc.sqc.createDataFrame(df_pd_train)
        C = rdd_instances.count()
        
        def __compute_part_obj_grad(part):
            lstSeq= []
            for row in part:
                lstSeq.append(row) 
            obj,grad = model.opt_callback_obj_grad(lstSeq)
            return [{"obj":obj,"grad":grad}]

        
        def __add(d1,d2):
            return {"obj":d1["obj"]+d2["obj"], "grad":d1["grad"]+d2["grad"]}

        def __obj(x):
            model.opt_callback_set_params(x)
            d = rdd_instances.mapPartitions(__compute_part_obj_grad,preservesPartitioning=True)\
            .reduce(lambda a,b:__add(a,b))
            #print type(d["grad"]/C)
            #print d["obj"]/C, d["grad"]/C
            print d["obj"]/C
            return d["obj"]/C, d["grad"]/C


        bnds = model.opt_callback_get_bounds()
        x0   = model.opt_callback_get_init_x()

        res = optimize.minimize(__obj,jac=True, x0= x0,method="L-BFGS-B",bounds=bnds,tol=tol,options={"disp":True})

        model.opt_callback_set_params(res.x)
        