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
    loss_=np.zeros(epochs)
    for epoch in range(epochs):
        Mloss = 0
        for batch, mc_vals,_,_ in train_loader:
            batch=batch.to(device)
            mc_vals=mc_vals.to(device)
            x1=mc_vals[:,0]
            x2=mc_vals[:,1]

            pred=model(batch).view(-1)
            loss = torch.mean(pred**2-x1*pred-x2*pred+x1*x2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            Mloss += loss
        print(f'[{epoch+1}/{epochs}] loss {Mloss/len(train_loader)}')
        loss_[epoch] = Mloss/len(train_loader)

    end_time=t.time()
    exe_time=end_time-start_time
    print("Training time: ",exe_time,"s")
    plt.plot(range(epochs), loss_)
    plt.xlabel('Epoch')
    plt.ylabel('Mean squared error')
    plt.yscale('log')
    plt.plot()


def eval(model, test_loader,device):
    model=model.to(device)
    with torch.no_grad():
        inf_time=0
        for batch,mc_vals,time_batch,nested_val in test_loader:
            
            time_batch=time_batch.numpy()
            nested_val=nested_val.numpy()
            start_time2 = t.time()
            pred=model(batch).view(-1).numpy()
            inf_time += t.time() - start_time2
            batch=batch.numpy()
            x1=mc_vals[:,0].numpy()
            x2=mc_vals[:,1].numpy()

            f1 = compute_Fvalue(x1, time_batch)
            f2 = compute_Fvalue(x2, time_batch)
            pred_f=compute_Fvalue(pred,time_batch)
            loss = np.abs(pred_f-nested_val)
            mse = np.mean(pred_f**2-f1*pred_f-f2*pred_f+f1*f2)

            plt.figure()
            _,_,_ = plt.hist(loss, 100, facecolor='g', alpha=0.75)
            plt.grid(True)
            plt.yscale('log')
            plt.show()
            plt.savefig("reg_histo.png")

            plt.figure()
            plt.scatter(time_batch, batch[:,0], c=pred_f, s=1, cmap='seismic')
            plt.colorbar()
            plt.show()
            plt.savefig("reg_traj.png")

            loss=np.mean(loss)
        print(f'Test loss: {loss}, Test MSE: {mse}, inference time {inf_time/(len(train_loader)*batch.shape[0])}')

   
        
#Load data from files
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

#Define Neural Network
device=torch.device('cuda')
model = nn.Linear(2,1).to(device)

#Define hyper-parameters
batch_size = 128
epochs = 25
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr)

#Prepare DataLoaders
x = torch.stack([price,i_t], dim=1).view(-1,2)
y = torch.stack((x1,x2),dim=1).view(-1,2)
dataset = torch.utils.data.TensorDataset(x,y,time,sum)
train_set,test_set = random_split(dataset,[len(dataset)-len(dataset)//7,len(dataset)//7])
train_loader=torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader=torch.utils.data.DataLoader(test_set, batch_size=len(test_set), shuffle=False)
print("Length of train set: %d\nLength of test set:  %d" %(len(train_set),len(test_set)))


#Training and evaluation
print('starting training...')
train(model, epochs, train_loader, optimizer,device)
torch.save(model.state_dict(),"reg_weights.pth")
print('training done.')
print('Evaluation...')
device=torch.device('cpu')
eval(model,test_loader,device)
