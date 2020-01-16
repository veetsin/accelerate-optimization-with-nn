import numpy as np
import torch
from torch.utils import data
import torch.nn.init as init
import sys
sys.path.append('/media/veetsin/qwerty/Projects/pytorch/opt_min')
import utils
from scipy.optimize import fmin_cobyla
from scipy.stats import ortho_group
import time 
from scipy.optimize import minimize


k = 6  #users
n = 4  #antennas
p = 1e7

test_len = 1000
batch_size = 100

#generate data set
# test_set = utils.MyDataset_precoding(k, n, p, test_len)
# test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
test_loader = utils.get_0116test()

net = torch.load('/media/veetsin/qwerty/Projects/pytorch/opt_min/0115002net_params')  



avg_time_naive = []
avg_time_to = []

result_naive = []
result_to = []

accur = 1e-2

for i, datas in enumerate(test_loader, 0):
    cofficient, label = datas
    h = cofficient[0]
    a = cofficient[1]
    start_time_naive = time.time()

    for j in range(batch_size):
        hj = h[j].to('cpu').numpy()
        aj = a[j].to('cpu').numpy()

        hj_H = np.array(np.matrix(hj).getH())
        # w_mrt = hj/np.linalg.norm(hj)
        w_mrt = -np.ones((n,k))*10
        # w_zf = hj.dot( np.linalg.inv(hj_H.dot(hj)) )
        w_zf = np.ones((n,k))*10

        def f(w, h_H=hj_H, a=aj):#funtion d
            diag = np.diag(h_H.dot(w.reshape(n,k)))
            sinr = np.array([(diag[i])**2/(1 + sum(np.delete((diag)**2, i))) for i in range(k) ])
            return -sum(a*np.log2(1 + sinr))
            
        
        cons = ({'type': 'ineq',  'fun' : lambda x: np.array(p - sum([np.linalg.norm(x.T[i])**2 for i in range(k)]) )})           

        res_zf = minimize(f, w_zf, constraints=cons, method='SLSQP', tol= accur,options={'disp': False})
        result1 = f(res_zf.x)

        res_mrt = minimize(f, w_mrt,constraints=cons,  method='SLSQP',tol= accur,options={'disp':False})
        result2 = f(res_mrt.x)
        result = min(result1, result2)
        
        result_naive.append(np.round(result, 2))


    time_naive= time.time() - start_time_naive
    avg_time_naive.append(round(time_naive, 2))
    

    start_time_to = time.time()
    outputs = net(cofficient[2].float()).reshape((batch_size, n, k))
    outputs = outputs.detach()
    for j in range(batch_size):
        hj = h[j].to('cpu').numpy()
        aj = a[j].to('cpu').numpy()
        wj = outputs[j].to('cpu').numpy()
        #1221001
        norm_sum =   sum([   np.linalg.norm((wj).T[i])**2 for i in range(k)   ]) 
        if norm_sum > p:
            wj = wj*np.sqrt(p/norm_sum)
        else:
            pass
    
        hj_H = np.array(np.matrix(hj).getH())

        def f(w, h_H=hj_H, a=aj):#funtion d
            diag = np.diag(h_H.dot(w.reshape(n,k)))
            sinr = np.array([(diag[i])**2/(1 + sum(np.delete((diag)**2, i))) for i in range(k) ])
            return -sum(a*np.log2(1 + sinr))
            
        
        cons = ({'type': 'ineq',  'fun' : lambda x: np.array(p - sum([np.linalg.norm(x.T[i]) for i in range(k)]) )})           

        res_to = minimize(f, wj, constraints=cons, method='SLSQP',tol=accur, options={'disp':False})
        
        result = f(res_to.x)
        result_to.append(np.round(result, 2))
        
    time_to = time.time() - start_time_to
    avg_time_to.append(round(time_to, 2))
        

print('time of naive optimizer(random) and transfer optimizing:')
print(avg_time_naive, np.mean(avg_time_naive), '\n', avg_time_to, np.mean(avg_time_to),'\n')

print('result of native optimize(better of random) and transfer optimizing: %.2f, %.2f' % (np.mean(result_naive), np.mean(result_to)) )



##100 0105002
# time of naive optimizer(zf+mrt) and transfer optimizing:
# [7.58, 7.71, 7.49, 7.85, 7.97, 7.73, 7.84, 7.77, 7.66, 7.81] 7.741
#  [3.82, 3.84, 3.64, 4.03, 3.94, 3.9, 3.91, 3.8, 3.87, 3.77] 3.852

# result of native optimize(better of zf and mrt) and transfer optimizing: -38.63, -36.65
# result of native optimize(better of zf and mrt) and transfer optimizing: -37.82, -36.17
##

##0305003
# time of naive optimizer(zf+mrt) and transfer optimizing:
# [1.54, 1.55, 1.56, 1.6, 1.54] 1.558
#  [0.77, 0.7, 0.8, 0.79, 0.77] 0.766

# result of native optimize(better of zf and mrt) and transfer optimizing: -37.40, -36.08



#1e-7:
# time of naive optimizer(random) and transfer optimizing:
# [73.89, 73.82, 75.08, 73.49, 77.82, 77.67, 74.29, 75.88, 74.38, 76.1] 75.242
#  [15.43, 15.16, 14.8, 15.81, 15.76, 14.77, 14.45, 14.6, 14.23, 14.95] 14.995999999999999
# result of native optimize(better of random) and transfer optimizing: -46.84, -49.58

# 1e-6:
# time of naive optimizer(random) and transfer optimizing:
# [48.77, 50.39, 50.15, 50.59, 48.21, 51.66, 50.54, 50.69, 49.77, 49.61] 50.038
#  [13.0, 13.59, 13.17, 13.12, 13.87, 13.83, 13.38, 13.92, 13.65, 13.86] 13.538999999999998

# result of native optimize(better of random) and transfer optimizing: -39.10, -47.97

#1e-5
# time of naive optimizer(random) and transfer optimizing:
# [32.39, 34.33, 33.33, 32.77, 31.75, 32.8, 33.51, 32.75, 34.33, 35.12] 33.308
#  [6.53, 6.22, 6.43, 6.37, 5.63, 6.01, 6.35, 6.26, 6.45, 8.25] 6.45

# result of native optimize(better of random) and transfer optimizing: -29.53, -31.99

# 1e-4
# time of naive optimizer(random) and transfer optimizing:
# [31.45, 31.72, 31.23, 31.48, 32.38, 31.38, 31.32, 31.32, 31.76, 32.0] 31.604000000000003
#  [3.16, 3.05, 3.43, 3.0, 2.84, 3.64, 3.22, 3.63, 3.48, 3.03] 3.2479999999999998

# result of native optimize(better of random) and transfer optimizing: -28.62, -24.88

# 1e-3
# time of naive optimizer(random) and transfer optimizing:
# [17.29, 21.89, 22.32, 18.27, 19.05, 19.85, 18.13, 17.86, 19.95, 20.17] 19.478
#  [1.73, 1.54, 1.7, 1.55, 1.74, 1.93, 2.01, 1.75, 2.06, 1.92] 1.793

# result of native optimize(better of random) and transfer optimizing: -19.21, -22.18