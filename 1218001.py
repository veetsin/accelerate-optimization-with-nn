#1218001 use constraint data as training set train 10000  test 1000 serious overfitting generate 1000 set need 10min
#1218002 generate 1000 every traing epoch, others same
#1218003 change size to 100 

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

n = 100
m = 1
epochs = 300
# epochs = 5

data_len = 2000
test_len = 500
batch_size = 100
log_loss = []

logging.basicConfig(level=logging.DEBUG,filename='/media/veetsin/qwerty/Projects/pytorch/opt_min/1218003.log',
    filemode='w',format='[%(levelname)s:%(message)s]')
start = time.time()
train_set = utils.MyDataset_constraint(n, m, 0.3, data_len)
data_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
print(time.time() - start)
test_set = utils.MyDataset_constraint(n, m, 0.3, test_len)
test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=True)


# net = utils.fc(n*n + n*m, 4096, 2048, 1024, n*m).cuda()
net = utils.fc(n*n + n*m, 2048, 2048, 1024, n*m).cuda()

l2loss = utils.L2NomrLoss()
optimizer = optim.Adam(net.parameters(), lr= 1e-5)
# optimizer = optim.LBFGS(net.parameters(), lr = 0.01)
# optimizer = optim.SGD(net.parameters(), lr=1e-5, momentum=0.9)


start = time.time()
for epoch in range(epochs):  # loop over the dataset multiple times-
    utils.adjust_learning_rate(optimizer,epoch)
    # train_set = utils.MyDataset_constraint(n, m, 0.3, data_len)
    # data_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
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

path = '/media/veetsin/qwerty/Projects/pytorch/opt_min/1218003net_params'
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