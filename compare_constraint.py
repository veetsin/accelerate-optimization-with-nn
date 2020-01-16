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


test_len = 10
batch_size = 2
mod = 0.3

#generate data set
test_set = utils.MyDataset_constraint_np(n, m, mod, test_len)
test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=True)


net = torch.load('/media/veetsin/qwerty/Projects/pytorch/opt_min/1221001net_params')  


x = torch.ones((n,m), requires_grad=True)

avg_time_cobyla = []
avg_time_to = []

result_cobyla = []
result_to = []

for i, datas in enumerate(test_loader, 0):
    cofficient, label = datas
    a = cofficient[0]
    b = cofficient[1]
    start_time_cobyla = time.time()
    for j in range(batch_size):
        aj = a[j].numpy()
        bj = b[j].numpy()
        break
    break

        def f(x, x1=aj, x2=bj):#funtion 
            return 1/2 * x.T.dot(x1).dot(x) + x2.T.dot(x)
            
        def constr(cofficient):
            return mod - np.linalg.norm(cofficient[0])

        x_opt_disp3_max1e5_1e3 x_cobyla = fmin_cobyla(f, np.zeros(n), [constr], rhoend=1e-7,maxfun=100000, disp=3)
        
        
         def func_deriv(x, x1=aj, x2=bj):
             return x.T.dot(x1) + x2.T
         

#        cons = ({'type': 'ineq',  'fun' : lambda x: np.array(mod - np.linalg.norm(x)), 'jac' : lambda x: np.array(2*x)})
        cons = ({'type': 'ineq',  'fun' : lambda x: np.array(mod - np.linalg.norm(x)) })

        
#        res = minimize(f, np.zeros(n), args=(-1.0,), jac=func_deriv,constraints=cons, method='SLSQP', options={'disp': True})
        res = minimize(f, np.zeros(n),constraints=cons, method='SLSQP', options={'disp': True})

        
        
        start = time.time()
        x_appxopt_acc1em1 = fmin_cobyla(f, np.ones(n)*10, [constr], rhoend=1,maxfun=10000, disp=3)
        x_appxopt_acc1em7 = fmin_cobyla(f, x_appxopt_acc1em1, [constr], rhoend=1e-3,maxfun=100000, disp=3)
        print(time.time() - start)
        
        
        
        
        
        # if i == (test_len/batch_size - 1) and j == (batch_size - 1):
        #     print(x_opt)
        result = f(x_opt)
        result_cobyla.append(np.round(result, 2))


    time_cobyla = time.time() - start_time_cobyla
    avg_time_cobyla.append(round(time_cobyla, 2))
    

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
        def f(x, x1=aj, x2=bj):#funtion 
            return 1/2 * x.T.dot(x1).dot(x) + x2.T.dot(x)
            
        def constr(cofficient):
            return mod - np.linalg.norm(cofficient[0])

        x_opt = fmin_cobyla(f, xj, [constr], rhoend=1e-7)
        result = f(x_opt)
        result_to.append(np.round(result, 2))
        # if i == (test_len/batch_size - 1) and j == (batch_size - 1):
        x_to = fmin_cobyla(f, xj, [constr], rhoend=1,maxfun=100000, disp=3)
        x_to1 = fmin_cobyla(f, x_to, [constr], rhoend=1e-3,maxfun=100000, disp=3)
        #     print(xj.reshape(50))
        #     print(x_opt.reshape(50))
    time_to = time.time() - start_time_to
    avg_time_to.append(round(time_to, 2))
        

print('time of naive optimizer and transfer optimizing:')
print(avg_time_cobyla, np.mean(avg_time_cobyla), '\n', avg_time_to, np.mean(avg_time_to),'\n')

print('result of native optimizer and transfer optimizing: %.2f, %.2f' % (np.mean(result_cobyla), np.mean(result_to)) )


# time of naive optimizer, and transfer optimizing: size 50
# [27.35, 27.37, 27.32, 27.29, 27.34, 27.3, 27.41, 28.16, 29.52, 29.53] 27.859
#  [27.37, 27.39, 27.37, 27.38, 27.42, 27.35, 27.42, 29.67, 29.4, 28.79] 27.956


#time of naive optimizer, and transfer optimizing:
#[66.03, 65.39, 63.96, 61.26, 61.25, 66.57, 67.11, 66.85, 66.69, 67.25] 65.23599999999999
#[65.73, 65.57, 61.32, 61.37, 63.5, 66.7, 66.88, 66.42, 66.42, 65.96] 64.987



#trained nn with conditional statement net 1221001 size 50
# time of naive optimizer, and transfer optimizing:
# [9.8, 9.65, 9.8, 9.84, 9.83, 9.71, 9.81, 9.83, 9.85, 9.68] 9.780000000000001
#  [9.57, 9.85, 9.86, 9.89, 9.73, 9.79, 9.82, 9.89, 9.88, 9.61] 9.789


# time of naive optimizer, and transfer optimizing:
# [9.58, 9.67, 9.34, 9.77, 9.75, 9.7, 9.83, 9.61, 9.87, 9.72] 9.684000000000001
#  [9.81, 9.49, 9.43, 9.85, 9.77, 9.76, 9.71, 9.75, 9.72, 9.41] 9.669999999999998

# [9.41, 9.43, 9.32, 9.34, 9.43, 9.35, 9.41, 9.55, 9.58, 9.39] 9.421000000000001
#  [9.44, 9.37, 9.37, 9.27, 9.35, 9.52, 9.35, 9.45, 9.46, 9.3] 9.388000000000002




#changed initial x0 to np.ones*99999999 print x_opt and xj
#two times get different x_opt but time almost no difference

# x_opt = fmin_cobyla(f, np.zeros(n), [constr], rhoend=1e-7)
# >>> x_opt1 = fmin_cobyla(f, np.zeros(n), [constr], rhoend=1e-7)
# >>> x_opt == x_opt1
# array([ True,  True,  True,  True,  True,  True,  True,  True,  True,
#         True,  True,  True,  True,  True,  True,  True,  True,  True,
#         True,  True,  True,  True,  True,  True,  True,  True,  True,
#         True,  True,  True,  True,  True,  True,  True,  True,  True,
#         True,  True,  True,  True,  True,  True,  True,  True,  True,
#         True,  True,  True,  True,  True])
# >>> x_opt2 = fmin_cobyla(f, np.zeros(n), [constr], rhoend=1e-7)
# >>> x_opt1 == x_opt2
# array([ True,  True,  True,  True,  True,  True,  True,  True,  True,
#         True,  True,  True,  True,  True,  True,  True,  True,  True,
#         True,  True,  True,  True,  True,  True,  True,  True,  True,
#         True,  True,  True,  True,  True,  True,  True,  True,  True,
#         True,  True,  True,  True,  True,  True,  True,  True,  True,
#         True,  True,  True,  True,  True])
# >>> x_opt2 = fmin_cobyla(f, np.ones(n), [constr], rhoend=1e-7)
# >>> x_opt == x_xopt2
# >>> x_opt == x_opt2
# array([False, False, False, False, False, False, False, False, False,
#        False, False, False, False, False, False, False, False, False,
#        False, False, False, False, False, False, False, False, False,
#        False, False, False, False, False, False, False, False, False,
#        False, False, False, False, False, False, False, False, False,
#        False, False, False, False, False])

# >>> f(x_opt)
# array([-0.81225623])
# >>> f(x_opt2)
# array([-0.78570171])
# >>> x_opt4 = fmin_cobyla(f, np.ones(n)*10000, [constr], rhoend=1e-7)
# >>> f(x_opt4)
# array([5.29279317e+10])
# >>> x_opt5 = fmin_cobyla(f, np.ones(n)*100, [constr], rhoend=1e-7)
# >>> f(x_opt5)
# array([456904.25839017])


#optimize from np.ones
# time of naive optimizer and transfer optimizing:
# [9.59, 9.69, 9.61, 9.73, 9.65, 9.65, 9.7, 9.71, 9.71, 9.64] 9.668000000000003
#  [9.61, 9.63, 9.61, 9.75, 9.64, 9.69, 9.61, 9.93, 9.63, 9.62] 9.672

# result of native optimizer and transfer optimizing: -0.59, -0.72

# optimize from np.zeros
# time of naive optimizer and transfer optimizing:
# [9.77, 10.04, 9.7, 9.55, 9.92, 9.86, 10.04, 9.76, 9.66, 9.74] 9.803999999999998
#  [10.29, 10.0, 9.72, 9.63, 9.95, 9.94, 9.98, 9.82, 9.68, 9.74] 9.874999999999998

# result of native optimizer and transfer optimizing: -0.75, -0.75

# x_opt:
# [ 0.03787247  0.09166698 -0.1359623  -0.04091571 -0.02562413 -0.05023666
#  -0.06452223 -0.0234247   0.0009928   0.07548254 -0.08842032  0.06214747
#   0.01120557 -0.05618934 -0.15285667 -0.10102463 -0.10836082 -0.13287502
#  -0.02328405 -0.11823641 -0.09128386 -0.01601153 -0.01968698 -0.10396845
#  -0.05979929 -0.00270558 -0.04693738  0.06180845 -0.18334017 -0.10531808
#   0.06930999 -0.0760254  -0.01292415 -0.04065085 -0.05898642 -0.10863722
#  -0.11755713 -0.06374969  0.02429516 -0.06522189  0.03084799 -0.07768267
#  -0.08389539 -0.07214435  0.01540347 -0.08288274 -0.12588408 -0.02172534
#  -0.00775889  0.042374  ]


# optimize from 10s
#time of naive optimizer and transfer optimizing:
# [9.24, 9.21, 9.2, 9.25, 9.24, 9.23, 9.21, 9.2, 9.25, 9.25] 9.228
#  [9.05, 9.07, 9.08, 9.05, 9.05, 9.07, 9.08, 9.09, 9.04, 9.09] 9.067000000000002

# result of native optimizer and transfer optimizing: 17.56, -0.74