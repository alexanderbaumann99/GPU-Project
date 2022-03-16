import torch
import numpy as np
from torch import nn
from os.path import join
from auxiliary import compute_Fvalue
import matplotlib.pyplot as plt
import time as t
from torch.utils.data import random_split


def train(model, epochs, train_loader, optimizer,device):
    start_time=t.time()
    for epoch in range(epochs):
        Mloss = 0
        for batch, target,_ in train_loader:
            batch=batch.to(device)
            input=batch[:,0:2]
            target=target.to(device).view(-1,2)
            x1=target[:,0]
            x2=target[:,1]

            pred=model(input).view(-1)
            loss = torch.mean(pred**2-x1*pred-x2*pred+x1*x2)

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
            input=batch[:,0:2]
            time_batch=batch[:,2].cpu().numpy()
            pred=model(input).view(-1).cpu().detach().numpy()
            pred_f=compute_Fvalue(pred,time_batch)

            target=target.cpu().numpy()
            loss = np.abs(pred_f-target)

            _,_,_ = plt.hist(loss, 100, facecolor='g', alpha=0.75)
            plt.grid(True)
            plt.yscale('log')
            plt.show()

            loss=np.mean(loss)

            Mloss += loss
        print(f'Test loss: {Mloss/len(test_loader)}')

    return Mloss

    
file_dir = join('./', 'nestedMC_data2') # does it work on windows?
time = np.loadtxt(join(file_dir, 'time_c.txt'), delimiter=',', usecols=1)
price = torch.FloatTensor(np.loadtxt(join(file_dir, 'price_c.txt'), delimiter=',', usecols=1))
i_t = torch.FloatTensor(np.loadtxt(join(file_dir, 'i_t_c.txt'), delimiter=',', usecols=1))
x1 = np.loadtxt(join(file_dir, 'x1_c.txt'), delimiter=',', usecols=1)
x2 = np.loadtxt(join(file_dir, 'x2_c.txt'), delimiter=',', usecols=1)
f1=compute_Fvalue(x1,time)
f2=compute_Fvalue(x2,time)
f1=torch.FloatTensor(f1)
f2=torch.FloatTensor(f2)
x1=torch.FloatTensor(x1)
x2=torch.FloatTensor(x2)
time=torch.FloatTensor(time)
sum = torch.FloatTensor(np.loadtxt(join(file_dir, 'sum_c.txt'), delimiter=',', usecols=1))


### regression on x'es
batch_size = 128
epochs = 20
lr = 0.001
hidden_dim=1024
device=torch.device('cuda')
model = nn.Sequential(  nn.Linear(2,hidden_dim),
                        nn.Linear(hidden_dim,1)).to(device)


x = torch.stack([price,i_t,time], dim=1).view(-1,3)
y = torch.stack((x1,x2),dim=1).view(-1,2)
dataset = torch.utils.data.TensorDataset(x,y,sum)
split=0.75
train_set,test_set=random_split(dataset,[len(dataset)-len(dataset)//4,len(dataset)//4])
train_loader=torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader=torch.utils.data.DataLoader(test_set, batch_size=len(test_set), shuffle=False)

optimizer = torch.optim.Adam(model.parameters(), lr)
model.load_state_dict(torch.load("regression_weights.pth"))
print('starting training...')
train(model, epochs, train_loader, optimizer,device)
print('training done.')
#torch.save(model.state_dict(),"regression_weights.pth")
print('Evaluation...')
device=torch.device('cpu')
_=eval(model,test_loader,device)

