import numpy as np
from os.path import join
import torch

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
        return self.augment(x).dot(self.weighs)

    def loss(self, x, y):
        return np.sum(np.square(self.predict(x)-y))


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