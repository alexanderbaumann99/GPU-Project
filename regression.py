import numpy as np
from os.path import join
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



class LinearRegression:
    def __init__(self, p):
        self.weights = np.zeros(p+1) # +1 for bias

    def augment(self, x):
        n, _ = x.shape
        x = np.c_[x, np.ones(n)]
        return x

    def train(self, x, y):
        # ich hab im internet gesehen, dass das mit
        # der QR decomposition eine schneller methode 
        # ist als die pseudo invers zu berechnen
        # keine ahnung ob das funktioniert
        # aber ist hier vll auch gar nicht so wichtig
        # wie evaluieren wir die regression eigentlich
        # ich denke, wir vergleichen einfach die varianz?
        q, r = np.linalg.qr(self.augment(x))
        self.weights = np.linalg.inv(r)@q.T.dot(y)

    def predict(self, x):
        return self.augment(x).dot(self.weights)

    def loss(self, x, y):
        return np.sum(np.square(self.predict(x)-y))

    def compute_value(self,x,t):

        prediction=self.predict(x)
        q=timeIdx(1.0)
        disc_pred=np.exp(-rt_int(t,1.0,0,q))*prediction

        return disc_pred




file_dir = join('.', 'nestedMC_data1') # does it work on windows?
time = np.loadtxt(join(file_dir, 'time_c.txt'), delimiter=',', usecols=1)
price = np.loadtxt(join(file_dir, 'price_c.txt'), delimiter=',', usecols=1)
i_t = np.loadtxt(join(file_dir, 'i_t_c.txt'), delimiter=',', usecols=1)
x1 = np.loadtxt(join(file_dir, 'x1_c.txt'), delimiter=',', usecols=1)
x2 = np.loadtxt(join(file_dir, 'x2_c.txt'), delimiter=',', usecols=1)

### regression
regression = LinearRegression(2)
x = np.c_[price, i_t]
print('training regression...')
regression.train(x, x1)
print('training done.')
print('regression lost', regression.loss(x, x1))
