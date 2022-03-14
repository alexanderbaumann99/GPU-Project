import torch
import numpy as np
from torch import nn
from os.path import join
from auxiliary import compute_Fvalue
import matplotlib.pyplot as plt

class NeuralNet(nn.Module):
    def __init__(self, dim_in, dim_out, dim_h, activation):
        super().__init__()
        layers = []
        dim_cur = dim_in
        for dim_next in dim_h:
            layers.append(nn.Linear(dim_cur, dim_next))
            layers.append(activation)
            dim_cur = dim_next
        layers.append(nn.Linear(dim_cur, dim_out))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x.float())

def train(model, epochs, lossFun, loader, optimizer):
    for epoch in range(epochs):
        Mloss = 0
        for batch, target in loader:
            loss = lossFun(model(batch), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            Mloss += loss
        print(f'[{epoch+1}/{epochs}] loss {Mloss/len(loader)}')
        
            


file_dir = join('.', 'nestedMC_data2') # does it work on windows?
time = np.loadtxt(join(file_dir, 'time_c.txt'), delimiter=',', usecols=1)
price = torch.from_numpy(np.loadtxt(join(file_dir, 'price_c.txt'), delimiter=',', usecols=1))
i_t = torch.from_numpy(np.loadtxt(join(file_dir, 'i_t_c.txt'), delimiter=',', usecols=1))
x1 = torch.from_numpy(np.loadtxt(join(file_dir, 'x1_c.txt'), delimiter=',', usecols=1))
x2 = torch.from_numpy(np.loadtxt(join(file_dir, 'x2_c.txt'), delimiter=',', usecols=1))
sum = np.loadtxt(join(file_dir, 'sum_c.txt'), delimiter=',', usecols=1)


### regression on x'es
batch_size = 128
epochs = 10
lr = 0.001
lossFun = nn.MSELoss()
model = NeuralNet(2, 1, [100, 100], nn.ReLU())

x = torch.stack([price, i_t], dim=1)
x = torch.cat([x, x], dim=0)
y = torch.cat([x1, x2], axis=0)
dataset = torch.utils.data.TensorDataset(x, y.unsqueeze(1).float())
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr)
print('starting training...')
train(model, epochs, lossFun, loader, optimizer)
print('training done.')

### bias of function values
x = torch.stack([price, i_t], dim=1)
pred = model(x).squeeze().detach().numpy()
Fpred = compute_Fvalue(pred, time)
bias = np.mean(np.abs(Fpred - sum))
print('Regression mean absolute bias:', bias)


### variance of function values
# i interpret the formula like this: 
# is this correct?
# get function values at X1 and X2
F1 = compute_Fvalue(x1.detach().numpy(), time)
F2 = compute_Fvalue(x2.detach().numpy(), time)
# predict values and get corresp F values
variance = np.mean(np.square(Fpred))-np.mean(Fpred*F1)-np.mean(Fpred*F2)+np.mean(F1*F2)
print('Regression mean variance', variance)


### visualization
plt.scatter(time, price, c=Fpred, s=1, cmap='seismic')
plt.colorbar()
plt.show()