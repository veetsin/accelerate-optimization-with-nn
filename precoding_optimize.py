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
p = 15 #power limit

#h = np.random.randint(1,10,size = (k,n))
h = np.random.normal(size=(n,k)) 
h_H = np.array(np.matrix(h).getH())

w_mrt = h/np.linalg.norm(h)
w_zf = h.dot( np.linalg.inv(h_H.dot(h)) )

a = np.random.rand(k)
a = np.array([round(i,2) for i in a])
avg_time_slsqp = []
# avg_time_to = []

result_slsqp = []
# result_to = []
t = 0
norm = []
for _ in range(50):
    h = np.random.normal(size=(n,k))
    h_H = np.array(np.matrix(h).getH())
    while np.linalg.det(h_H.dot(h)) == 0:
        h = np.random.normal(size=(n,k))
        h_H = np.array(np.matrix(h).getH())


    w_mrt = h/np.linalg.norm(h)
    w_zf = h.dot( np.linalg.inv(h_H.dot(h)) )

    a = np.random.rand(k)
    a = np.array([round(i,2) for i in a])

    def f(w, h_H=h_H, a=a):#funtion d
        # print(1)
        # print(h.shape)
        # print(w.shape)
        # print(np.diag(h.dot(w.reshape(4,6))))
        diag = np.diag(h_H.dot(w.reshape(n,k)))
        # s = 0
        # for i in range(k):
        #     sinr_i = np.abs(diag[i])**2/(1 + sum(np.delete( np.abs(diag)**2, i)))
        #     s = s + (-a[i]*np.log2(1+sinr_i) )
        #     print(s)
        # return s
        sinr = np.array([(diag[i])**2/(1 + sum(np.delete((diag)**2, i))) for i in range(k) ])
        return -sum(a*np.log2(1 + sinr))


    cons = ({'type': 'ineq',  'fun' : lambda x: np.array(p - sum([np.square(np.linalg.norm(x.T[i])) for i in range(k)]) )})           
    res = minimize(f, np.ones((n,k)), constraints=cons, method='SLSQP', tol= 1e-7,options={'disp': False})
    res_mrt = minimize(f, w_mrt, constraints=cons, method='SLSQP', tol= 1e-7,options={'disp': False})
    res_zf = minimize(f, w_zf, constraints=cons, method='SLSQP', tol= 1e-7,options={'disp': False})
    # res_mrt = minimize(f, w_mrt, method='SLSQP', tol= 1e-7,options={'disp': False})
    # res_zf = minimize(f, w_zf,method='SLSQP', tol= 1e-7,options={'disp': False})
    if f(res_mrt.x) > f(res_zf.x) :
        t += 1
        norm_sum =  sum([np.square(np.linalg.norm(res_zf.x.T[i])) for i in range(k)]) 
        norm.append(norm_sum)
    else:
        norm_sum1 = sum([np.square(np.linalg.norm(res_mrt.x.T[i])) for i in range(k)])
        norm.append(norm_sum1)
    
print(t)
print(np.mean(norm))


# print(f(res.x))
# print(f(res_mrt.x))
# print(f(res_zf.x))
# result_slsqp.append(np.round(result, 2))


# time_slsqp= time.time() - start_time_cobyla
# avg_time_slsqp.append(round(time_slsqp, 2))


# >>> scipy.misc.derivative(f,np.ones((n,k)), n=2, dx= 1e-6 )
# 0.20816681711721685
# >>> scipy.misc.derivative(f,-np.ones((n,k)), n=2, dx= 1e-6 )
# 0.20816681711721685
# >>> scipy.misc.derivative(f,np.zeros((n,k)), n=2, dx= 1e-6 )
# -39.90746090031194


# >>> scipy.misc.derivative(f,np.zeros((n,k)), n=1, dx= 1e-6 )
# 0.0



# with mrt
# Optimization terminated successfully.    (Exit mode 0)
#             Current function value: -28.11925302958016
#             Iterations: 83
#             Function evaluations: 2184
#             Gradient evaluations: 83
# Optimization terminated successfully.    (Exit mode 0)
#             Current function value: -39.60999131611947
#             Iterations: 83
#             Function evaluations: 2213
#             Gradient evaluations: 83


# Optimization terminated successfully.    (Exit mode 0)
#             Current function value: -42.93254292599235
#             Iterations: 87
#             Function evaluations: 2303
#             Gradient evaluations: 87
# Optimization terminated successfully.    (Exit mode 0)
#             Current function value: -46.78238572548347
#             Iterations: 83
#             Function evaluations: 2192
#             Gradient evaluations: 83


#zf and mrt 
#-16.88462580569997
# -18.915827539073106
# -42.586529619076416



#for 100 samples 41 mrt better than zf