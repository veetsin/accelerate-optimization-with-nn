import numpy as np
import torch
from torch.utils import data
import torch.nn.init as init
import logging
import torch.optim as optim
from torch import nn 
from scipy.stats import ortho_group
from scipy.optimize import fmin_cobyla
from scipy.optimize import minimize
import pickle


n = 50
m = 1

def f(x, a, b):#funtion 
    return 1/2 * x.T.dot(a).dot(x) + b.T.dot(x) 

def f_torch(x, a, b):#funtion 
    xta = torch.mm(torch.transpose(x,0,1),a)
    return 1/2 * torch.mm(xta, x) + torch.mm(torch.transpose(b,0,1), x)

#generate orthonormal matirx
def a_pd_inv(a, diag): #a positive definite and invertible
    a = torch.mm(torch.mm(a, diag.double()), torch.transpose(a,0,1))
    # a = a.dot(a.T)
    while np.abs(np.linalg.det(a.to('cpu') )) < 1e-3:
        a = ortho_group.rvs(n).cuda()
        a = torch.mm(torch.mm(a, diag.double()), torch.transpose(a,0,1))
        # a = np.random.random((n,n))
        # a = a.dot(a.T)
    return a

def a_pd_inv_np(a, diag): #a positive definite and invertible
    a = a.dot(diag).dot(a.T)
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

#generate data set

class MyDataset(data.Dataset):

    def __init__(self, n, m, data_len):
        data_list = []
        for _ in range(data_len):
            a = ortho_group.rvs(n)
            a = torch.from_numpy(a).cuda()
            diag = np.diag(np.random.randint(1, 50, n))
            diag = torch.from_numpy(diag).cuda()
            a = a_pd_inv(a, diag)


            b = np.random.random((n, m))
            x_opt = -np.linalg.inv(a.to('cpu')).dot(b)
            fx_opt = f_torch(torch.from_numpy(x_opt).cuda(), a, torch.from_numpy(b).cuda())
            ab = np.append(a.to('cpu').reshape((n*n)), b.reshape((n*m)) )
            ab = torch.from_numpy(ab).cuda()
            b = torch.from_numpy(b).cuda()
            # x_opt = torch.from_numpy(x_opt)
            fx_opt = fx_opt.cuda()
            data_list.append([a, b, ab, fx_opt])
        self.data_list = data_list#a,b, a+b, f(x_opt)
        
    
    def __getitem__(self, index):
        cofficient, label = self.data_list[index][0:3], self.data_list[index][-1].reshape((m, m))
        return cofficient, label
    

    def __len__(self):
        return len(self.data_list)




class MyDataset_constraint(data.Dataset):

    def __init__(self, n, m, mod, data_len):#0.3 mod 
        data_list = []
        for _ in range(data_len):
            a = ortho_group.rvs(n)
            diag = np.diag(np.random.randint(1, 50, n))
            a = a_pd_inv_np(a, diag)
            b = np.random.random((n, m))
            
            def f(x, x1=a, x2=b):#funtion 
                return 1/2 * x.T.dot(x1).dot(x) + x2.T.dot(x)
            
            def constr(cofficient):
                return mod - np.linalg.norm(cofficient[0])
            
            x_opt = fmin_cobyla(f, np.zeros(n), [constr], rhoend=1e-7)
            fx_opt = f(x_opt)
            ab = np.append(a.reshape((n*n)), b.reshape((n*m)) )
            ab = torch.from_numpy(ab).cuda()
            b = torch.from_numpy(b).cuda()
            # x_opt = torch.from_numpy(x_opt)
            fx_opt = torch.from_numpy(fx_opt).cuda()
            a = torch.from_numpy(a).cuda()
            data_list.append([a, b, ab, fx_opt])
        
        self.data_list = data_list#a,b, a+b, f(x_opt)
    

    def __getitem__(self, index):
        cofficient, label = self.data_list[index][0:3], self.data_list[index][-1].reshape((m, m))
        return cofficient, label
    

    def __len__(self):
        return len(self.data_list)

class MyDataset_constraint_slsqp(data.Dataset):

    def __init__(self, n, m, mod, data_len):#0.3 mod 
        data_list = []
        for _ in range(data_len):
            a = ortho_group.rvs(n)
            diag = np.diag(np.random.randint(1, 50, n))
            a = a_pd_inv_np(a, diag)
            b = np.random.random((n, m))
            
            def f(x, x1=a, x2=b):#funtion 
                return 1/2 * x.T.dot(x1).dot(x) + x2.T.dot(x)

            cons = ({'type': 'ineq',  'fun' : lambda x: np.array(mod - np.linalg.norm(x)) })
            
            res = minimize(f, np.ones(n), constraints=cons, method='SLSQP', options={'disp': False})
            fx_opt = f(res.x)
            ab = np.append(a.reshape((n*n)), b.reshape((n*m)) )
            ab = torch.from_numpy(ab).cuda()
            b = torch.from_numpy(b).cuda()
            # x_opt = torch.from_numpy(x_opt)
            fx_opt = torch.from_numpy(fx_opt).cuda()
            a = torch.from_numpy(a).cuda()
            data_list.append([a, b, ab, fx_opt])
        
        self.data_list = data_list#a,b, a+b, f(x_opt)
    

class MyDataset_constraint_np_slsqp(data.Dataset):

    def __init__(self, n, m, mod, data_len):#0.3 mod 
        data_list = []
        for _ in range(data_len):
            a = ortho_group.rvs(n)
            diag = np.diag(np.random.randint(1, 50, n))
            a = a_pd_inv_np(a, diag)
            b = np.random.random((n, m))
            
            def f(x, x1=a, x2=b):#funtion 
                return 1/2 * x.T.dot(x1).dot(x) + x2.T.dot(x)

            cons = ({'type': 'ineq',  'fun' : lambda x: np.array(mod - np.linalg.norm(x)) })
            
            res = minimize(f, np.ones(n), constraints=cons, method='SLSQP', options={'disp': False})
            fx_opt = f(res.x)
            ab = np.append(a.reshape((n*n)), b.reshape((n*m)) )
            ab = torch.from_numpy(ab).cuda()
            # b = torch.from_numpy(b).cuda()
            # x_opt = torch.from_numpy(x_opt)
            # fx_opt = torch.from_numpy(fx_opt).cuda()
            # a = torch.from_numpy(a).cuda()
            data_list.append([a, b, ab, fx_opt])
        
        self.data_list = data_list#a,b, a+b, f(x_opt)

    def __getitem__(self, index):
        cofficient, label = self.data_list[index][0:3], self.data_list[index][-1].reshape((m, m))
        return cofficient, label
    

    def __len__(self):
        return len(self.data_list)
    

class MyDataset_constraint_np(data.Dataset):

    def __init__(self, n, m, mod, data_len):#0.3 mod 
        data_list = []
        for _ in range(data_len):
            a = ortho_group.rvs(n)
            diag = np.diag(np.random.randint(1, 50, n))
            a = a_pd_inv_np(a, diag)
            b = np.random.random((n, m))
            
            def f(x, x1=a, x2=b):#funtion 
                return 1/2 * x.T.dot(x1).dot(x) + x2.T.dot(x)
            
            def constr(cofficient):
                return mod - np.linalg.norm(cofficient[0])
            
            x_opt = fmin_cobyla(f, np.zeros(n), [constr], rhoend=1e-7)
            fx_opt = f(x_opt)
            ab = np.append(a.reshape((n*n)), b.reshape((n*m)) )
            ab = torch.from_numpy(ab).cuda()
            # b = torch.from_numpy(b).cuda()
            # x_opt = torch.from_numpy(x_opt)
            # fx_opt = torch.from_numpy(fx_opt).cuda()
            # a = torch.from_numpy(a).cuda()
            data_list.append([a, b, ab, fx_opt])
        
        self.data_list = data_list#a,b, a+b, f(x_opt)
    

    def __getitem__(self, index):
        cofficient, label = self.data_list[index][0:3], self.data_list[index][-1].reshape((m, m))
        return cofficient, label
    

    def __len__(self):
        return len(self.data_list)





class MyDataset_precoding(data.Dataset):

    def __init__(self, k, n, p, data_len):#0.3 mod 
        data_list = []
        for _ in range(data_len):
            h = np.random.normal(size=(n,k))
            h_H = np.array(np.matrix(h).getH())
            while np.linalg.det(h_H.dot(h)) == 0:
                h = np.random.normal(size=(n,k))
                h_H = np.array(np.matrix(h).getH())
            w_mrt = h/np.linalg.norm(h)
            w_zf = h.dot( np.linalg.inv(h_H.dot(h)) )

#            def get_w_zf(h):
#                h = h+1e-8j
#                h_H = np.array(np.matrix(h).getH())
#                w_zf = h.dot( np.linalg.inv(h_H.dot(h)) )
#                return np.real(w_zf)
#            w_zf = get_w_zf(h)                

            a = np.ones(k)
            def f_precoding(w, h_H=h_H, a=a):#funtion d
                diag = np.diag(h_H.dot(w.reshape(n,k)))
                sinr = np.array([(diag[i])**2/(1 + sum(np.delete((diag)**2, i))) for i in range(k) ])
                return -sum(a*np.log2(1 + sinr))

            cons = ({'type': 'ineq',  'fun' : lambda x: np.array(p - sum([ np.linalg.norm(x.T[i])**2 for i in range(k)]) )})           
            # res = minimize(f_precoding, np.ones((n,k)), constraints=cons, method='SLSQP', tol=1e-7, options={'disp': False})
            res_mrt = minimize(f_precoding, w_mrt, constraints=cons, method='SLSQP', tol= 1e-7,options={'disp': False})
            res_zf = minimize(f_precoding, w_zf, constraints=cons, method='SLSQP', tol= 1e-7,options={'disp': False})
            # fx_opt_ones = f_precoding(res.x)
            fx_opt_mrt = f_precoding(res_mrt.x)
            fx_opt_zf = f_precoding(res_zf.x)
            # fx_opt = np.min([fx_opt_ones,fx_opt_mrt,fx_opt_zf])
            fx_opt = np.min([fx_opt_mrt,fx_opt_zf])

            ha = np.append(h.reshape((k*n)), a.reshape((k*1)) )
            h = torch.from_numpy(h).cuda()
            a = torch.from_numpy(a).cuda()
            ha = torch.from_numpy(ha).cuda()
            fx_opt = torch.from_numpy(np.array(fx_opt)).cuda()
            data_list.append([h, a, ha, fx_opt])
        
        self.data_list = data_list#h, a, ha, f(x_opt)
    

    def __getitem__(self, index):
        cofficient, label = self.data_list[index][0:3], self.data_list[index][-1]
        return cofficient, label
    

    def __len__(self):
        return len(self.data_list)




class MyDataset_precoding_targetx(data.Dataset):

    def __init__(self, k, n, p, data_len):#0.3 mod 
        data_list = []
        for _ in range(data_len):
            h = np.random.normal(size=(n,k))
            h_H = np.array(np.matrix(h).getH())
            while np.linalg.det(h_H.dot(h)) == 0:
                h = np.random.normal(size=(n,k))
                h_H = np.array(np.matrix(h).getH())
            w_mrt = h/np.linalg.norm(h)
            w_zf = h.dot( np.linalg.inv(h_H.dot(h)) )

#            def get_w_zf(h):
#                h = h+1e-8j
#                h_H = np.array(np.matrix(h).getH())
#                w_zf = h.dot( np.linalg.inv(h_H.dot(h)) )
#                return np.real(w_zf)
#            w_zf = get_w_zf(h)                

            a = np.ones(k)
            def f_precoding(w, h_H=h_H, a=a):#funtion d
                diag = np.diag(h_H.dot(w.reshape(n,k)))
                sinr = np.array([(diag[i])**2/(1 + sum(np.delete((diag)**2, i))) for i in range(k) ])
                return -sum(a*np.log2(1 + sinr))

            cons = ({'type': 'ineq',  'fun' : lambda x: np.array(p - sum([np.linalg.norm(x.T[i]) for i in range(k)]) )})           
            # res = minimize(f_precoding, np.ones((n,k)), constraints=cons, method='SLSQP', tol=1e-7, options={'disp': False})
            res_mrt = minimize(f_precoding, w_mrt, constraints=cons, method='SLSQP', tol= 1e-7,options={'disp': False})
            res_zf = minimize(f_precoding, w_zf, constraints=cons, method='SLSQP', tol= 1e-7,options={'disp': False})
            # fx_opt_ones = f_precoding(res.x)
            fx_opt_mrt = f_precoding(res_mrt.x)
            fx_opt_zf = f_precoding(res_zf.x)
            
            if fx_opt_zf < fx_opt_mrt:
                x_opt = res_zf.x
            else:
                x_opt = res_mrt.x
            # fx_opt = np.min([fx_opt_ones,fx_opt_mrt,fx_opt_zf])
            ha = np.append(h.reshape((k*n)), a.reshape((k*1)) )
            h = torch.from_numpy(h).cuda()
            a = torch.from_numpy(a).cuda()
            ha = torch.from_numpy(ha).cuda()
            x_opt = torch.from_numpy(np.array(x_opt)).cuda()
            data_list.append([h, a, ha, x_opt])
        
        self.data_list = data_list#h, a, ha, x_opt
    

    def __getitem__(self, index):
        cofficient, label = self.data_list[index][0:3], self.data_list[index][-1]
        return cofficient, label
    

    def __len__(self):
        return len(self.data_list)
    
    
class fc(nn.Module):

    def __init__(self, in_dim, n_hidden_1, n_hidden_2, n_hidden_3, out_dim):
        super(fc, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2), nn.ReLU(True))
        # self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3), nn.BatchNorm1d(n_hidden_3), nn.ReLU(True))
        self.layer4 = nn.Sequential(nn.Linear(n_hidden_3, out_dim))
 
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

class fc_precoding(nn.Module):

    def __init__(self, in_dim, n_hidden_1, n_hidden_2, n_hidden_3, n_hidden_4, n_hidden_5, n_hidden_6, out_dim):
        super(fc_precoding, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2), nn.ReLU(True))
        # self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3), nn.BatchNorm1d(n_hidden_3), nn.ReLU(True))
        self.layer4 = nn.Sequential(nn.Linear(n_hidden_3, n_hidden_4), nn.BatchNorm1d(n_hidden_4), nn.ReLU(True))
        self.layer5 = nn.Sequential(nn.Linear(n_hidden_4, n_hidden_5), nn.BatchNorm1d(n_hidden_5), nn.ReLU(True))
        self.layer6 = nn.Sequential(nn.Linear(n_hidden_5, n_hidden_6), nn.BatchNorm1d(n_hidden_6), nn.ReLU(True))

        self.layer7 = nn.Sequential(nn.Linear(n_hidden_6, out_dim))
 
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x) 
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)

        return x


class L2NomrLoss(nn.Module):
    def __init__(self):
        super(L2NomrLoss, self).__init__()


    def forward(self, out, target):
        # loss = torch.from_numpy(np.linalg.norm(target - out))  #euclidean distance
        # loss_all = torch.norm((((out - target))/target).float())
        loss_all = torch.abs((((out - target))/target).float())
        loss = torch.mean(loss_all)
        return loss

class FNormLoss(nn.Module):
    def __init__(self):
        super(FNormLoss, self).__init__()

    def forward(self, out, target):
        loss  = torch.norm(out-target).float()
        return torch.mean(loss)

def adjust_learning_rate(optimizer, epoch):
    if epoch == 64:
        lr = 1e-6
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    if epoch == 128:
        lr = 1e-7
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    if epoch == 192:
        lr = 1e-8
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    if epoch == 256:
        lr = 1e-9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr    

def adjust_learning_rate_precoding(optimizer, epoch):
    start_lr = 3
    if epoch == 50:#256
        lr = start_lr/3
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    if epoch == 100:#512
        lr = start_lr/10
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    if epoch == 150:#768
        lr = start_lr/30
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    if epoch == 200:#1024
        lr = start_lr/100
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr    

def adjust_learning_rate_precoding_targetx(optimizer, epoch):
    if epoch == 256:
        lr = 20
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    if epoch == 512:
        lr = 10
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    if epoch == 768:
        lr = 5
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    if epoch == 1024:
        lr = 1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr    


def get_0105test():
    f = open('/media/veetsin/qwerty/Projects/pytorch/opt_min/0105test.txt','rb')
    test_loader = pickle.load(f)
    f.close()
    return test_loader


def get_0105train():
    f = open('/media/veetsin/qwerty/Projects/pytorch/opt_min/0105.txt','rb')
    train_loader = pickle.load(f)
    f.close()
    return train_loader

def get_12210114test():
    f = open('/media/veetsin/qwerty/Projects/pytorch/opt_min/12210114.txt','rb')
    test_loader = pickle.load(f)
    f.close()
    return test_loader


def get_0115test():
    f = open('/media/veetsin/qwerty/Projects/pytorch/opt_min/0115test.txt','rb')
    test_loader = pickle.load(f)
    f.close()
    return test_loader


def get_0115train():
    f = open('/media/veetsin/qwerty/Projects/pytorch/opt_min/0115train.txt','rb')
    train_loader = pickle.load(f)
    f.close()
    return train_loader

def get_0116test():
    f = open('/media/veetsin/qwerty/Projects/pytorch/opt_min/0116test.txt','rb')
    test_loader = pickle.load(f)
    f.close()
    return test_loader


def get_0116train():
    f = open('/media/veetsin/qwerty/Projects/pytorch/opt_min/0116train.txt','rb')
    train_loader = pickle.load(f)
    f.close()
    return train_loader