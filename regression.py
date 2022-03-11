import numpy as np
from os.path import join

file_dir = join('.', 'nestedMC_data1') # does it work on windows?
time = np.loadtxt(join(file_dir, 'time_c.txt'), delimiter=',', usecols=1)
price = np.loadtxt(join(file_dir, 'price_c.txt'), delimiter=',', usecols=1)
i_t = np.loadtxt(join(file_dir, 'i_t_c.txt'), delimiter=',', usecols=1)
x1 = np.loadtxt(join(file_dir, 'x1_c.txt'), delimiter=',', usecols=1)
x2 = np.loadtxt(join(file_dir, 'x2_c.txt'), delimiter=',', usecols=1)