import numpy as np
import torch

#copy from MC.cu
def params():
    
        Tgc=np.empty((15,))
        rgc=np.empty((15,))
            
        d = 1/360
        w = 7.0 * d
        m = 30. * d
        y = 12. * m
        
        Tgc[0] = d
        Tgc[1] = 2*d
        Tgc[2] = w
        Tgc[3] = 2.*w
        Tgc[4] = m
        Tgc[5] = 2*m
        Tgc[6] = 3*m
        Tgc[7] = 6*m
        Tgc[8] = y
        Tgc[9] = y + 3*m
        Tgc[10] = y + 6*m
        Tgc[11] = 2*y
        Tgc[12] = 2*y + 6*m
        Tgc[13] = 3*y
        Tgc[14] = 3*y + 6.*m

        rgc[0] = 0.05
        rgc[1] = 0.07
        rgc[2] = 0.08
        rgc[3] = 0.06
        rgc[4] = 0.07
        rgc[5] = 0.1
        rgc[6] = 0.11
        rgc[7] = 0.13
        rgc[8] = 0.12
        rgc[9] = 0.14
        rgc[10] = 0.145
        rgc[11] = 0.14
        rgc[12] = 0.135
        rgc[13] = 0.13
        rgc[14] = 0.*y

        return Tgc,rgc


Tg,rg=params()

#copy from MC.cu
def rt_int(t,T,i,j):

        if(i==j):
            res = (T-t)*rg[i]
        else:
            res = (T-Tg[j-1])*rg[j] + (Tg[i]-t)*rg[i]
                        
            for k in range(i+1,j):
                res += (Tg[k]-Tg[k-1])*rg[k]

        return res

#copy from MC.cu
def timeIdx(t):
     
        for i in range(14,-1,-1):
            if(t<Tg[i]):
                I = i           
            
        return I

def compute_Fvalue(x,t):
    q=timeIdx(1.0)
    disc_pred=np.exp(-rt_int(t,1.0,0,q))*x
    return disc_pred

def compute_Fvalue2(x,t):
    q=timeIdx(1.0)
    disc_pred=np.exp(-2*rt_int(t,1.0,0,q))*x
    return disc_pred