#add bathch_max_difference use np.random.randint to generate diagnol matrix 1212001
#test loss 1212002  
#use abs instead of norm 1212003
#if BN layer works 1212004
import numpy as np
import torch
from torch.utils import data
import torch.nn.init as init

n = 50
m = 1
epochs = 256
# epochs = 5

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

# t1=torch.tensor([[1,2,3],[2,3,4]])
# t2=torch.tensor([[1,2,3],[2,3,4]])
# t3=torch.tensor([[111,2,3],[222,3,4]])
# l=[]
# l.append(t1)
# l.append(t2)
# l.append(t3)


data_len = 3000
test_len = 100
batch_size = 10
log_loss = []

#generate orthonormal matirx
import logging 
logging.basicConfig(level=logging.DEBUG,filename='/media/veetsin/qwerty/Projects/pytorch/opt_min/1212004.log',
    filemode='w',format='[%(levelname)s:%(message)s]')

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

train_set = MyDataset(n, m, data_len)
test_set = MyDataset(n, m, test_len)
data_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=True)


import torch.optim as optim
from torch import nn

class fc(nn.Module):

    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(fc, self).__init__()
        # self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1), nn.ReLU(True))
        # self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2), nn.ReLU(True))
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))
 
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x



net = fc(n*n + n*m, 1024, 512 ,n*m)

class L2NomrLoss(nn.Module):
    def __init__(self):
        super(L2NomrLoss, self).__init__()


    def forward(self, out, target):
        # loss = torch.from_numpy(np.linalg.norm(target - out))  #euclidean distance
        # loss_all = torch.norm((((out - target))/target).float())
        loss_all = torch.abs((((out - target))/target).float())
        loss = torch.mean(loss_all)
        return loss

l2loss = L2NomrLoss()
# optimizer = optim.Adam(net.parameters(), lr=0.000001)
# optimizer = optim.LBFGS(net.parameters(), lr = 0.01)
optimizer = optim.SGD(net.parameters(), lr=1e-5, momentum=0.9)

def adjust_learning_rate(optimizer, epoch):
    if epoch == 100:
        lr = 3e-6
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    if epoch == 150:
        lr = 3e-7
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    if epoch == 200:
        lr = 1e-7
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    

for epoch in range(epochs):  # loop over the dataset multiple times-
    for i, datas in enumerate(data_loader, 0):
    #     break
    # break
        # get the inputs; data is a list of [inputs, labels]
        # zero the parameter gradients
        # def closure():
        optimizer.zero_grad()
        cofficient, label = datas
        outputs = net(cofficient[2].float()) #20*600
        a_list = cofficient[0]
        b_list = cofficient[1]
    #     break
    # break
        fx = []
        for j in range(batch_size):
            x = outputs[j].reshape((n,m)).float()
            a = a_list[j].float()
            b = b_list[j].float()
            tem = f_torch(x, a, b)
            fx.append(tem)
        fx = torch.stack(fx).float()
        label = label.float()
        # print(fx)
        # print(label)
        loss = l2loss(fx, label)
        loss.backward()
        # return loss

        # loss = closure()
        # optimizer.step(closure)
        optimizer.step()
        # break

        # print statistics
        loss = loss.item()
        if i % 10 == 9:    
            print('[%d, %5d] loss: %.3f' %  (epoch + 1, i + 1, loss))
            logging.info('[%d, %5d] loss: %.3f' %  (epoch + 1, i + 1, loss))
            log_loss.append(loss)

print('Finished Training')


from matplotlib import pyplot as plt 
log_loss = [i for i in log_loss if i < 1000 ]
x_axis = np.linspace(0,len(log_loss),len(log_loss))
plt.figure(figsize=(20,15))
plt.plot(x_axis,log_loss,label='loss')
plt.xlabel('epoch')
plt.legend()
plt.show()


avg_max = []

log_difference= []

for i, datas in enumerate(test_loader, 0):
    cofficient, label = datas
    a = cofficient[0].float()
    b = cofficient[1].float()
    outputs = net(cofficient[2].float()).reshape((10, 50, 1))
    out = []
    for j in range(len(outputs)):
        out.append(f_torch(outputs[j],a[j],b[j]))
    out = torch.stack(out).double()
    loss = torch.mean((out - label))
    test_loss = torch.abs(torch.mean((out - label)/label))
    batch_max = (out - label).max().detach().numpy()
    avg_max.append(batch_max)

    log_difference.append(loss.detach().numpy())

  
    print('test batch: %d, difference: %.3f batch_max_difference: %.3f, test loss: %.3f' %  (i + 1, loss.detach().numpy() , batch_max, test_loss ))
    logging.info('test batch: %d, difference: %.3f batch_max_difference: %.3f, test loss: %.3f' %  (i + 1, loss.detach().numpy() , batch_max,test_loss ))



print('average test difference: %.3f, average max difference: %.3f, max difference: %.3f' % ( sum(log_difference)/len(log_difference), sum(avg_max)/len(avg_max), max(avg_max)))
logging.info('average test difference: %.3f, average max difference: %.3f, max difference: %.3f' % ( sum(log_difference)/len(log_difference), sum(avg_max)/len(avg_max), max(avg_max)))

# path = '/media/veetsin/qwerty/Projects/pytorch/opt_min/qua_func_opt_1212001net_params'
# torch.save(net, path)




x = torch.ones((n,m), requires_grad=True)
optimizer_naive = torch.optim.Adam([x],lr=1e-2)
import time
start = time.time()
for step in range(1000):
    # def closure():
    pred = f_torch(x.float(),a.float(),b.float())
    optimizer_naive.zero_grad()
    pred.backward()
    print(pred)
    optimizer_naive.step()
end = time.time()
print(end - start)

#LBFGS: 48s for 1000epochs , lr 1e-4; 3.507s for 50 epochs and converges lr 0.01
# 
# Adam:2.56s  -0.546
# net = torch.load('/media/veetsin/qwerty/Projects/pytorch/opt_min/1208002net_params')  
