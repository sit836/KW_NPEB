# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 16:50:19 2018

@author: billt
"""

import matplotlib.pyplot as plt
import mosek
import numpy as np
import sys
from scipy.stats import norm

def _streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()
    
def kwp(y,m1,m2): 
    lenGrid = len(m1)   
    n = len(y)
    location = np.subtract.outer(y,m1)
    A = np.asarray([norm.pdf(location[i],scale=np.sqrt(m2)) for i in range(n)])
    # A[A < 1e-12] = 1e-12    
    
    A1 = np.concatenate((A,-np.identity(n)),axis=1)
    A2 = np.array([1]*lenGrid + [0]*n).T    
    A = np.vstack((A1,A2))
    
    with mosek.Env() as env:
        env.set_Stream(mosek.streamtype.log, _streamprinter)
        with env.Task(0, 0) as task:
            task.set_Stream(mosek.streamtype.log, _streamprinter)
            # task.putdouparam(mosek.dparam.intpnt_co_tol_rel_gap, 1.0e-8)

            numvar = n + lenGrid
            numcon = n + 1
            # Since the actual value of Infinity is ignores, we define it solely
            # for symbolic purposes:

            bkc = [mosek.boundkey.fx]*numcon
            blc = [0]*n + [1]
            buc = [0]*n + [1]

            bkx = [mosek.boundkey.ra]*numvar
            blx = [0]*numvar
            bux = [1]*numvar

            task.appendvars(numvar)
            task.appendcons(numcon)
            
            task.putvarboundslice(0, numvar, bkx, blx, bux)
            task.putconboundslice(0, numcon, bkc, blc, buc)
            
            opro = [mosek.scopr.log]*n
            oprjo = list(range(lenGrid,numvar))
            oprfo = [-1]*n
            oprgo = [1]*n
            oprho = [0]*n
                    
            asub = [list(range(A.shape[0]))]*numvar
            aval = []
            # aval[j] contains the non-zero values of column j
            for i in range(0,A.shape[1]):
                aval.append(list(A[:,i]))  
                
            for j in range(numvar):    
                task.putacol(j,asub[j],aval[j])     
  
                
            oprc = [mosek.scopr.ent]
            opric = [0]
            oprjc = [0]
            oprfc = [0.0]
            oprgc = [0.0]
            oprhc = [0.0]

            task.putSCeval(opro, oprjo, oprfo, oprgo, oprho,
                           oprc, opric, oprjc, oprfc, oprgc, oprhc)        

            task.optimize()
                       
            v = [0.0] * numvar
            task.getsolutionslice(
                mosek.soltype.itr,
                mosek.solitem.xx,
                0, numvar,
                v)

#==============================================================================
#             task.putintparam(
#                 mosek.iparam.write_ignore_incompatible_items, mosek.onoffkey.on)
#==============================================================================
            
            p = v[0:lenGrid]
            mixture = v[lenGrid:numvar]
            
            return {'p':p,'mixture':mixture}
            
def npeb_prediction(y,prior,m1,m2):  
    location = np.subtract.outer(y,m1)
    A = np.asarray([norm.pdf(location[i],scale=np.sqrt(m2)) for i in range(len(y))])

    weighted_support = np.matmul(np.diag(prior),np.c_[m1,m2])    
    pred = np.matmul(np.diag(1/np.matmul(A,prior)),np.matmul(A,weighted_support))

    return pred            
            
def plot_constraints(y,dualSol,m1,m2): 
    location = np.subtract.outer(y,m1)
    A = np.asarray([norm.pdf(location[i],scale=np.sqrt(m2)) for i in range(len(y))])

    con = np.matmul(A.transpose(),dualSol)
    plt.plot(range(len(con)),con)            
    

          
      
            
            
            
            
            
            
            
            
            
            
            