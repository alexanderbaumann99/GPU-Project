import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from auxiliary import compute_Fvalue

class LinearRegression:
    def __init__(self, p):
        self.weights = np.zeros(p+1) # +1 for bias

    def augment(self, x):
        n, _ = x.shape
        x = np.c_[x, np.ones(n)]
        return x

    def train(self, x, y):
        q, r = np.linalg.qr(self.augment(x))
        self.weights = np.linalg.solve(r, q.T.dot(y))

    def predict(self, x):
        return self.augment(x).dot(self.weights)

    def loss(self, x, y):
        n, _ = x.shape
        return np.mean(np.square(self.predict(x)-y))


file_dir = join('.', 'nestedMC_data2') # does it work on windows?
time = np.loadtxt(join(file_dir, 'time_c.txt'), delimiter=',', usecols=1)
price = np.loadtxt(join(file_dir, 'price_c.txt'), delimiter=',', usecols=1)
i_t = np.loadtxt(join(file_dir, 'i_t_c.txt'), delimiter=',', usecols=1)
x1 = np.loadtxt(join(file_dir, 'x1_c.txt'), delimiter=',', usecols=1)
x2 = np.loadtxt(join(file_dir, 'x2_c.txt'), delimiter=',', usecols=1)
sum = np.loadtxt(join(file_dir, 'sum_c.txt'), delimiter=',', usecols=1)

### regression on x'es
regression = LinearRegression(2)
x = np.c_[price, i_t]
x = np.concatenate([x, x], axis=0)
y = np.concatenate([x1, x2], axis=0)
print('regression lost', regression.loss(x, y))
print('training regression...')
regression.train(x, y)
print('training done.')
print('regression loss', regression.loss(x, y))

### bias of function values
x = np.c_[price, i_t]
pred = regression.predict(x)
Fpred = compute_Fvalue(pred, time)
bias = np.mean(np.abs(Fpred - sum))
print('Regression mean absolute bias:', bias)


### variance of function values
# i interpret the formula like this: 
# is this correct?
# get function values at X1 and X2
F1 = compute_Fvalue(x1, time)
F2 = compute_Fvalue(x2, time)
# predict values and get corresp F values
variance = np.mean(np.square(Fpred))-np.mean(Fpred*F1)-np.mean(Fpred*F2)+np.mean(F1*F2)
print('Regression mean variance', variance)


### visualization
plt.scatter(time, price, c=Fpred, s=1, cmap='seismic')
plt.colorbar()
plt.show()
