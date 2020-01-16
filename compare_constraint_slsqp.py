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


n = 50
m = 1


test_len = 1000
batch_size = 100
mod = 0.3

#generate data set
t =time.time()
test_loader = utils.get_12210114test()
print(time.time() -t)
toler = 1e-6

net = torch.load('/media/veetsin/qwerty/Projects/pytorch/opt_min/12210114net_params')  


x = torch.ones((n,m), requires_grad=True)

avg_time_slsqp = []
avg_time_to = []

result_slsqp = []
result_to = []

for i, datas in enumerate(test_loader, 0):
    cofficient, label = datas
    a = cofficient[0]
    b = cofficient[1]
    start_time_cobyla = time.time()
    for j in range(batch_size):
        aj = a[j].numpy()
        bj = b[j].numpy()


        def f(x, x1=aj, x2=bj):#funtion 
            return 1/2 * x.T.dot(x1).dot(x) + x2.T.dot(x)
            
        
        cons = ({'type': 'ineq',  'fun' : lambda x: np.array(mod - np.linalg.norm(x)) })
#        res = minimize(f, np.ones(n)*100,constraints=cons, method='SLSQP', options={'disp': False})
#        t1=time.time()
        res = minimize(f, np.ones(n)*100,constraints=cons, method='SLSQP' ,tol = toler ,options={'disp': False})
#        print(time.time()-t1)
#        print(f(res.x))
##        start = time.time()
        result = f(res.x)
#        t2=time.time()
#        res1 = minimize(f, np.ones(n), constraints=cons,method='COBYLA',tol=1e-7,options={'disp':True,'maxiter':100000})
#        print(time.time()-t2)
#        print(f(res1.x))
        result_slsqp.append(np.round(result, 2))


    time_slsqp= time.time() - start_time_cobyla
    avg_time_slsqp.append(round(time_slsqp, 2))
    

    start_time_to = time.time()
    outputs = net(cofficient[2].float()).reshape((batch_size, n, m))
    outputs = outputs.detach()
    for j in range(batch_size):
        aj = a[j].numpy()
        bj = b[j].numpy()
        norm = torch.norm(outputs[j])
        #1221001
        if norm > mod:
            xj = (outputs[j]/norm*mod).to('cpu').numpy()
        else:
            xj = outputs[j].to('cpu').numpy()
        # xj = outputs[j].to('cpu').numpy()
        def f1(x, x1=aj, x2=bj):#funtion 
            return 1/2 * x.T.dot(x1).dot(x) + x2.T.dot(x)
            
        cons = ({'type': 'ineq',  'fun' : lambda x: np.array(mod - np.linalg.norm(x)) })
        res_to = minimize(f1, xj , constraints=cons, method='SLSQP', tol = toler, options={'disp': False })
        
        result = f1(res_to.x)
        result_to.append(np.round(result, 2))
        
    time_to = time.time() - start_time_to
    avg_time_to.append(round(time_to, 2))
        

print('time of naive optimizer and transfer optimizing for every batch:')
print(avg_time_slsqp, np.mean(avg_time_slsqp), '\n', avg_time_to, np.mean(avg_time_to),'\n')

print('average result of native optimizer and transfer optimizing: %.2f, %.2f' % (np.mean(result_slsqp), np.mean(result_to)) )




#from ones 
# time of naive optimizer and transfer optimizing:
# [6.43, 6.39, 6.29, 6.24, 6.32, 6.38, 6.2, 6.32, 6.08, 6.26] 6.2909999999999995
#  [6.14, 6.07, 5.87, 5.89, 5.92, 5.92, 5.8, 5.79, 5.65, 5.87] 5.8919999999999995

# result of naive optimizer and transfer optimizing: -0.57, -0.57

#from 10s
# [6.81, 7.03, 6.62, 6.83, 6.86, 6.78, 6.72, 6.95, 6.7, 6.62] 6.792
#  [6.3, 6.32, 6.04, 6.43, 6.28, 6.07, 6.26, 6.28, 5.97, 5.99] 6.194000000000001

# result of naive optimizer and transfer optimizing: -0.57, -0.57


#optimize from different points with tolerance 1e-3
# >>> res = minimize(f, np.ones(n)*10,constraints=cons, method='SLSQP' ,tol = 1e-3, options={'disp': True})
# Optimization terminated successfully.    (Exit mode 0)
#             Current function value: -0.5162012682452294
#             Iterations: 52
#             Function evaluations: 2767
#             Gradient evaluations: 52
# >>> print(time.time()-t1)
# 0.07454180717468262

# >>> res_to = minimize(f, xj, constraints=cons, method='SLSQP', tol=1e-3, options={'disp': True })
# Optimization terminated successfully.    (Exit mode 0)
#             Current function value: -0.5125578969155862
#             Iterations: 14
#             Function evaluations: 745
#             Gradient evaluations: 14

#1000test 100batch 1e-1 -1e-7 from random.rand
#1e-7:
# time of naive optimizer and transfer optimizing for every batch:
# [6.07, 6.16, 6.23, 6.49, 6.17, 6.24, 6.2, 6.36, 6.25, 6.07] 6.224
#  [5.88, 5.93, 6.16, 6.08, 6.04, 6.04, 6.0, 6.22, 5.82, 5.87] 6.004

# average result of native optimizer and transfer optimizing: -0.58, -0.58
# 1e-6:
# time of naive optimizer and transfer optimizing for every batch:
# [6.46, 6.06, 6.21, 6.02, 6.22, 6.08, 6.03, 6.09, 6.01, 6.31] 6.149
#  [6.08, 5.74, 5.7, 5.77, 5.92, 5.59, 5.65, 5.63, 5.6, 5.87] 5.755

# average result of native optimizer and transfer optimizing: -0.58, -0.58

# 1e-5:
# time of naive optimizer and transfer optimizing for every batch:
# [6.17, 6.45, 5.97, 6.08, 6.27, 5.78, 5.69, 5.9, 6.08, 5.97] 6.036
#  [5.58, 5.49, 5.31, 5.38, 5.09, 4.83, 4.79, 5.02, 5.36, 5.29] 5.2139999999999995

# average result of native optimizer and transfer optimizing: -0.58, -0.58

# 1e-4:
# time of naive optimizer and transfer optimizing for every batch:
# [6.02, 5.99, 5.6, 5.8, 5.99, 5.92, 5.83, 5.97, 6.01, 5.86] 5.898999999999999
#  [4.28, 4.28, 3.92, 4.21, 4.19, 4.28, 4.26, 4.28, 4.49, 4.09] 4.228

# average result of native optimizer and transfer optimizing: -0.58, -0.58

# 1e-3:
# time of naive optimizer and transfer optimizing for every batch:
# [5.67, 5.35, 5.32, 5.33, 5.18, 5.13, 5.29, 5.34, 5.61, 5.27] 5.348999999999999
#  [3.0, 2.85, 2.55, 2.74, 2.63, 2.7, 2.68, 2.94, 2.9, 2.62] 2.761

# average result of native optimizer and transfer optimizing: -0.58, -0.58

# 1e-2:
# time of naive optimizer and transfer optimizing for every batch:
# [4.41, 4.03, 4.0, 4.09, 4.05, 4.03, 3.78, 3.93, 4.19, 4.06] 4.057
#  [1.38, 1.19, 1.08, 1.22, 1.12, 1.15, 1.05, 1.14, 1.03, 1.1] 1.1459999999999997

# average result of native optimizer and transfer optimizing: -0.58, -0.56

# 1e-1:
# time of naive optimizer and transfer optimizing for every batch:
# [2.15, 2.07, 2.03, 2.14, 2.21, 2.05, 2.06, 2.09, 2.05, 2.05] 2.0900000000000003
#  [0.31, 0.29, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.29, 0.29] 0.298

# average result of native optimizer and transfer optimizing: -0.61, -0.51

#1e-6 change start
#ones
# time of naive optimizer and transfer optimizing for every batch:
# [6.47, 6.29, 6.29, 6.21, 6.06, 6.19, 6.02, 6.25, 6.12, 6.18] 6.208
#  [5.9, 5.65, 5.9, 5.71, 5.61, 5.87, 5.63, 5.78, 5.56, 5.86] 5.747

# average result of native optimizer and transfer optimizing: -0.58, -0.58
#tens
#onehundred
#