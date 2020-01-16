#add bathch_max_difference use np.random.randint to generate diagnol matrix 1212001
#test loss 1212002  
#use abs instead of norm 1212003
#if BN layer works 1212004
#1216001use different training set when training (transfer data to gpu)
#1216002 net = utils.fc(n*n + n*m, 4096, 2048, 1024, n*m).cuda() train 50000 batch 100 
#1216003 net = utils.fc(n*n + n*m, 4096, 2048, 1024, n*m).cuda() train 50000 batch 100 epoch 300
#1216004 net = utils.fc(n*n + n*m, 4096, 2048, 1024, n*m).cuda() train 3000 every epoch batch 100 epoch 300 adam 
#1216005 train 2000 every batch 100 test 1000 256epoch 
#1216007 n = 50 generate 5000 sets of data every training epoch to get well trained nn net = utils.fc(n*n + n*m, 4096, 2048, 1024, n*m).cuda() 
#           batch 100 test 10000   300 epochs adam epochs 1e-5,   64,128,192,256, 1e-6 -7 -8 -9



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

n = 50
m = 1
epochs = 300
# epochs = 5

data_len = 50000
test_len = 10000
batch_size = 500
log_loss = []

logging.basicConfig(level=logging.DEBUG,filename='/media/veetsin/qwerty/Projects/pytorch/opt_min/1216007.log',
    filemode='w',format='[%(levelname)s:%(message)s]')

train_set = utils.MyDataset(n, m, data_len)
data_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_set = utils.MyDataset(n, m, test_len)
test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=True)


net = utils.fc(n*n + n*m, 4096, 2048, 1024, n*m).cuda()

l2loss = utils.L2NomrLoss()
optimizer = optim.Adam(net.parameters(), lr= 1e-5)
# optimizer = optim.LBFGS(net.parameters(), lr = 0.01)
# optimizer = optim.SGD(net.parameters(), lr=1e-5, momentum=0.9)


start = time.time()
for epoch in range(epochs):  # loop over the dataset multiple times-
    utils.adjust_learning_rate(optimizer,epoch)
    train_set = utils.MyDataset(n, m, data_len)
    data_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
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
            tem = utils.f_torch(x, a, b)
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
end = time.time()
print('Finished Training, time: %.3f' % (end-start))
logging.info('Finished Training, time: %.3f' % (end-start))


from matplotlib import pyplot as plt 
log_loss = [i for i in log_loss if i < 5 ]
x_axis = np.linspace(0,len(log_loss),len(log_loss))
plt.figure(figsize=(20,15))
plt.plot(x_axis,log_loss,label='loss')
plt.xlabel('epoch')
plt.legend()
plt.show()

path = '/media/veetsin/qwerty/Projects/pytorch/opt_min/1216007net_params'
torch.save(net, path)

avg_max = []

log_difference= []

for i, datas in enumerate(test_loader, 0):
    cofficient, label = datas
    a = cofficient[0].float()
    b = cofficient[1].float()
    outputs = net(cofficient[2].float()).reshape((batch_size, n, m))
    out = []
    for j in range(len(outputs)):
        out.append(utils.f_torch(outputs[j],a[j],b[j]))
    out = torch.stack(out).double()
    loss = torch.mean((out - label))
    test_loss = torch.abs(torch.mean((out - label)/label))
    batch_max = (out - label).to('cpu').max().detach().numpy()
    avg_max.append(batch_max)

    log_difference.append(loss.detach().to('cpu').numpy())

  
    print('test batch: %d, difference: %.3f batch_max_difference: %.3f, test loss: %.3f' %  (i + 1, loss.detach().to('cpu').numpy() , batch_max, test_loss ))
    logging.info('test batch: %d, difference: %.3f batch_max_difference: %.3f, test loss: %.3f' %  (i + 1, loss.detach().to('cpu').numpy() , batch_max,test_loss ))



print('average test difference: %.3f, average max difference: %.3f, max difference: %.3f' % ( sum(log_difference)/len(log_difference), sum(avg_max)/len(avg_max), max(avg_max)))
logging.info('average test difference: %.3f, average max difference: %.3f, max difference: %.3f' % ( sum(log_difference)/len(log_difference), sum(avg_max)/len(avg_max), max(avg_max)))






# x = torch.ones((n,m), requires_grad=True)
# # optimizer_naive = torch.optim.Adam([x],lr=1e-2)
# optimizer_naive = torch.optim.LBFGS([x],line_search_fn='strong_wolfe')
# import time
# start = time.time()
# for step in range(1000):
#     def closure():
#         pred= f_torch(x.float(),a.float(),b.float())
#         optimizer_naive.zero_grad()
#         pred.backward()
#         print(pred)
#         return pred
#     optimizer_naive.step(closure)
# end = time.time()
# print(end - start)

#LBFGS: 48s for 1000epochs , lr 1e-4; 3.507s for 50 epochs and converges lr 0.01
# 
# Adam:2.56s  -0.546
# net = torch.load('/media/veetsin/qwerty/Projects/pytorch/opt_min/1208002net_params')  

#   File "/home/veetsin/.conda/envs/pytorch/lib/python3.6/site-packages/torch/optim/lbfgs.py", line 205, in step
#     raise RuntimeError("line search function is not supported yet")
# RuntimeError: line search function is not supported yet
