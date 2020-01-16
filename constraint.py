import numpy as np
from scipy.optimize import fmin_cobyla
from scipy.stats import ortho_group


n = 100
m = 1

def a_pd_inv(a, diag): #a positive definite and invertible
    a = a.dot(diag).dot(a.T)
    # a = a.dot(a.T)
    while np.abs(np.linalg.det(a)) < 1e-3:
        a = ortho_group.rvs(n)
        a = a.dot(diag).dot(a.T) 
        # a = np.random.random((n,n))
        # a = a.dot(a.T)
    return a

a = ortho_group.rvs(n)
diag = np.diag(np.random.randint(1, 50, n))
a = a_pd_inv(a, diag)
b = np.random.random((n, m))
x_opt = -np.linalg.inv(a).dot(b)

mean_norm = []

for i in range(1000):
    a = ortho_group.rvs(n)
    diag = np.diag(np.random.randint(1, 50, n))
    a = a_pd_inv(a, diag)
    b = np.random.random((n, m))
    x_opt = -np.linalg.inv(a).dot(b)
    mean_norm.append(np.linalg.norm(x_opt))

print(np.mean(mean_norm))
#1000 norm 0.64 0.65
#n = 100, norm 1
# array([[-0.23342013],[ 0.07147005]])
# array([[-0.1       , -0.00168473]])
    
# fx_opt = f(x_opt, a, b) 

# def f(x, a=np.array([[4.44298142, 1.95032496],[1.95032496, 3.55701858]]), b=np.array([[0.89769147],[0.20102482]])):#funtion 
#     return 1/2 * x.T.dot(a).dot(x) + b.T.dot(x) 

def f(x, a=a, b=b):#funtion 
    return 1/2 * x.T.dot(a).dot(x) + b.T.dot(x)

def constr(cofficient):
   return 0.0001 - np.linalg.norm(cofficient[0])

import time
start = time.time()
x_opt_norm = fmin_cobyla(f, np.zeros(n), [constr], rhoend=1e-7)
# fmin = array([[-0.23342014,  0.07147002]])
