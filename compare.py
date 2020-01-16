import numpy as np
import torch
from torch.utils import data
import torch.nn.init as init
import sys
sys.path.append('/media/veetsin/qwerty/Projects/pytorch/opt_min')
import utils

n = 50
m = 1
training_epochs = [1000, 2000, 3000, 4000]
# training_epochs = [2, 3]

def f(x, a, b):#funtion 
    return 1/2 * x.T.dot(a).dot(x) + b.T.dot(x) 

def f_torch(x, a, b):#funtion 
    xta = torch.mm(torch.transpose(x,0,1),a)
    return 1/2 * torch.mm(xta, x) + torch.mm(torch.transpose(b,0,1), x)

def a_pd_inv(a, diag): #a positive definite and invertible
    a = a.dot(diag).dot(a.T)
    # a = a.dot(a.T)
    while np.abs(np.linalg.det(a)) < 1e-3:
        a = ortho_group.rvs(n)
        a = a.dot(diag).dot(a.T)
        # a = np.random.random((n,n))
        # a = a.dot(a.T)
    return a

test_len = 100
batch_size = 10


#generate data set
from scipy.stats import ortho_group
class MyDataset(data.Dataset):

    def __init__(self, n, m, data_len):
        data_list = []
        for _ in range(data_len):
            a = ortho_group.rvs(n)
            diag = np.diag(np.random.randint(1, 50, n))
            a = a_pd_inv(a, diag)
            b = np.random.random((n, m))
            x_opt = -np.linalg.inv(a).dot(b)
            fx_opt = f(x_opt, a, b)
            ab = np.append(a.reshape((n*n)), b.reshape((n*m)) )
            ab = torch.from_numpy(ab)
            a = torch.from_numpy(a)
            b = torch.from_numpy(b)
            # x_opt = torch.from_numpy(x_opt)
            fx_opt = torch.from_numpy(fx_opt)
            data_list.append([a, b, ab, fx_opt])
        self.data_list = data_list#a,b, a+b, f(x_opt)
        
    
    def __getitem__(self, index):
        cofficient, label = self.data_list[index][0:3], self.data_list[index][-1].reshape((m, m))
        return cofficient, label
    

    def __len__(self):
        return len(self.data_list)

test_set = MyDataset(n, m, test_len)
test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

import torch.optim as optim
from torch import nn

class fc(nn.Module):

    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(fc, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))
 
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x



# net = fc(n*n + n*m, 1024, 512 ,n*m)
# net = torch.load('/media/veetsin/qwerty/Projects/pytorch/opt_min/1216001net_params')  


x = torch.ones((n,m), requires_grad=True)

time_naive = []
time_nn = []
difference_nn = []
avg_time_naive = []
avg_time_nn = []
avg_difference_naive = []
avg_difference_nn = []
import time 
for i in range(len(training_epochs)):
    # if i == 0:
    #     start_time_nn = time.time()
    #     for j, datas in enumerate(test_loader, 0):
    #         # print(datas)
    #         # break
    #         cofficient, label = datas
    #         a = cofficient[0].float().cuda()
    #         b = cofficient[1].float().cuda()
    #         # outputs = net(cofficient[2].cuda().float()).reshape((10, 50, 1))
    #         outputs = net(cofficient[2].cuda().float())
    #         out = []
    #         for l in range(len(outputs)):
    #             out.append(f_torch(outputs[l],a[l],b[l]))
    #         out = torch.stack(out).double()
    #         difference = torch.mean( torch.abs(( (out.double() - label)) ))
    #         print(difference)
    #         difference_nn.append(difference.detach().numpy())
    #     end_time_nn = time.time()
    #     avg_difference_nn = np.mean(difference_nn)
    #     avg_time_nn.append(end_time_nn - start_time_nn)

    # else:
    #     pass


    difference_naive = []
    for j, datas in enumerate(test_loader, 0):
        cofficient, label = datas
        a = cofficient[0]
        b = cofficient[1]
                
        start_time_naive = time.time()
        x = torch.ones((batch_size, n, m), requires_grad=True)
        optimizer_naive = torch.optim.LBFGS([x],lr = 1e-3)
        for step in range(training_epochs[i]):
            def closure():
                fx = []
                for l in range(len(x)):
                    pred = f_torch(x[l].float(),a[l].float(),b[l].float())  
                    fx.append(pred)
                    fx = torch.stack(fx).float()
                    print(fx)
                    optimizer_naive.zero_grad()
                    fx.backward()
                    return fx
            out = optimizer_naive.step(closure)
    difference_naive.append( np.abs(out.detach().double() - label ).numpy())
    end_time_naive = time.time()

    avg_difference_naive.append(np.mean(difference_naive))
    avg_time_naive.append(end_time_naive - start_time_naive) 





print(avg_difference_naive, avg_difference_nn ,avg_time_naive,avg_time_nn)


# [51.023, 0.745, 0.237, 0.211][0.733] 

# [1.075, 2.153, 3.155, 3.612] [0.053]




# [12.188, 0.406, 0.2505, 0.284]  [0.523]

#  [8.072, 16.413, 24.632, 32.939]  [0.055]
