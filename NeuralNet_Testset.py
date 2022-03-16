import torch
import numpy as np
from torch import nn
from os.path import join
from auxiliary import compute_Fvalue
import matplotlib.pyplot as plt
import time as t
import math
from torch.utils.data import random_split

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

def train(model, epochs, train_loader, optimizer,device):
    start_time=t.time()
    for epoch in range(epochs):
        Mloss = 0
        for batch, target,_ in train_loader:
            batch=batch.to(device)
            target=target.to(device).view(-1,2)
            f1=target[:,0]
            f2=target[:,1]

            pred=model(batch).view(-1)
            loss = torch.mean(pred**2-f1*pred-f2*pred+f1*f2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            Mloss += loss
        print(f'[{epoch+1}/{epochs}] loss {Mloss/len(train_loader)}')

    end_time=t.time()
    exe_time=end_time-start_time
    print("Training time: ",exe_time,"s")

def eval(model, test_loader,device):
    model=model.to(device)
    with torch.no_grad():
        Mloss = 0
        for batch,_,target in test_loader:
            batch=batch.to(device).view(-1,3)
            
            pred=model(batch).view(-1).detach().numpy()
            target=target.cpu().numpy()

            loss = pred-target

            _,_,_ = plt.hist(loss, 100, facecolor='g', alpha=0.75)
            plt.grid(True)
            plt.yscale('log')
            plt.show()
            plt.savefig("nn_histo.png")

            loss=np.mean(np.abs(loss))

            Mloss += loss
        print(f'Test loss: {Mloss/len(test_loader)}')

    return Mloss
        
#file_dir = "/kaggle/input/gpu-neuralnet/nestedMC_data2"  
file_dir = join('./', 'nestedMC_data2') # does it work on windows?
price = torch.FloatTensor(np.loadtxt(join(file_dir, 'price_c.txt'), delimiter=',', usecols=1))
i_t = torch.FloatTensor(np.loadtxt(join(file_dir, 'i_t_c.txt'), delimiter=',', usecols=1))
sum = torch.FloatTensor(np.loadtxt(join(file_dir, 'sum_c.txt'), delimiter=',', usecols=1))
time = np.loadtxt(join(file_dir, 'time_c.txt'), delimiter=',', usecols=1)
x1 = np.loadtxt(join(file_dir, 'x1_c.txt'), delimiter=',', usecols=1)
x2 = np.loadtxt(join(file_dir, 'x2_c.txt'), delimiter=',', usecols=1)
f1=compute_Fvalue(x1,time)
f2=compute_Fvalue(x2,time)
f1=torch.FloatTensor(f1)
f2=torch.FloatTensor(f2)
x1=torch.FloatTensor(x1)
x2=torch.FloatTensor(x2)
time=torch.FloatTensor(time)

batch_size = 128
epochs = 20
lr = 0.001
lossFun = nn.MSELoss()
device=torch.device('cuda')

model = NeuralNet(3, 1, [128,64,32], nn.LeakyReLU()).to(device)

x = torch.stack([price,i_t,time], dim=1).view(-1,3)
y=torch.stack((f1,f2),dim=1).view(-1,2)
dataset = torch.utils.data.TensorDataset(x,y,sum)
train_set,test_set=random_split(dataset,[len(dataset)-len(dataset)//7,len(dataset)//7])
train_loader=torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader=torch.utils.data.DataLoader(test_set, batch_size=len(test_set), shuffle=False)
print("Length of train set: %d\nLength of test set:  %d" %(len(train_set),len(test_set)))
optimizer = torch.optim.Adam(model.parameters(), lr)
model.load_state_dict(torch.load("NN_weights.pth"))
print('starting training...')
#train(model, epochs, train_loader, optimizer,device)
torch.save(model.state_dict(),"NN_weights.pth")
print('training done.')
print('Evaluation...')
device=torch.device('cpu')
_=eval(model,test_loader,device)
