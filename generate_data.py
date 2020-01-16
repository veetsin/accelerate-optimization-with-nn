import pickle
import numpy as np
import torch
from torch.utils import data
import torch.nn.init as init
import logging
import torch.optim as optim
from torch import nn 
import sys
sys.path.append('/media/veetsin/qwerty/Projects/pytorch/opt_min')
import utils
import time
k = 6  #users
n = 4  #antennas
p = 1e7 #power limit

data_len = 5000
test_len = 1000
batch_size = 100

#
test_set = utils.MyDataset_precoding(k, n, p, test_len)
test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=True)


f = open('/media/veetsin/qwerty/Projects/pytorch/opt_min/0115test.txt','wb')
pickle.dump(test_loader,f)
f.close()
# f = open('/media/veetsin/qwerty/Projects/pytorch/opt_min/0105test.txt','wb')
# pickle.dump(test_loader,f)
# f.close()

#
train_set = utils.MyDataset_precoding(k, n, p, data_len)
data_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)


f = open('/media/veetsin/qwerty/Projects/pytorch/opt_min/0115train.txt','wb')
pickle.dump(data_loader,f)
f.close()
# f = open('/media/veetsin/qwerty/Projects/pytorch/opt_min/0105.txt','wb')
# pickle.dump(data_loader,f)
# f.close()

#12210114
n = 50
m = 1
test_len = 1000
batch_size = 100
mod = 0.3
test_set_12210114 = utils.MyDataset_constraint_np_slsqp(n, m, mod, test_len)
test_loader_12210114 = data.DataLoader(test_set_12210114, batch_size=batch_size, shuffle=True)

f = open('/media/veetsin/qwerty/Projects/pytorch/opt_min/12210114.txt','wb')
pickle.dump(test_loader_12210114,f)
f.close()