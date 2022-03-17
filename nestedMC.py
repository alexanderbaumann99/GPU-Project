import numpy as np
from os.path import join
import matplotlib.pyplot as plt

from auxiliary import compute_Fvalue2

Nouter = 8192
Ninner = 4096

### load data
file_dir = join('.', 'nestedMC_data2') # does it work on windows?
time = np.loadtxt(join(file_dir, 'time_c.txt'), delimiter=',', usecols=1)
price = np.loadtxt(join(file_dir, 'price_c.txt'), delimiter=',', usecols=1)
i_t = np.loadtxt(join(file_dir, 'i_t_c.txt'), delimiter=',', usecols=1)#.reshape((Nouter, -1))
sum = np.loadtxt(join(file_dir, 'sum_c.txt'), delimiter=',', usecols=1)
sum2 = np.loadtxt(join(file_dir, 'sum2_c.txt'), delimiter=',', usecols=1)

### calulate variance
print(sum)
print(sum2)
variance = compute_Fvalue2(sum2 - np.square(sum), time)/Ninner
print('Mean variance', np.mean(variance))

### visualize
plt.scatter(time, price, c=variance, s=1, cmap='seismic')
plt.colorbar()
plt.show()