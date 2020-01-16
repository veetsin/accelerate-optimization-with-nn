#1221001    add a conditional statement to help training the nn with constraint; 1000 new every epoch, 1000 test n = 50 m = 1 mod = 0.3 epoch300; 
#           net = utils.fc(n*n + n*m, 4096, 2048, 1024, n*m) optimizer = optim.Adam(net.parameters(), lr= 1e-5) 64 128 192 256 -6 -7 -8 -9

#1221002    use 50000 sets of data as training set others same 
#RuntimeError: CUDA out of memory. Tried to allocate 1024.00 KiB (GPU 0; 7.93 GiB total capacity; 6.72 GiB already allocated; 24.75 MiB free; 10.50 KiB cached)
#buchong: 12210114 same as 1221001 1000every new 1000test
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
mod = 0.3
epochs = 300
path = '/media/veetsin/qwerty/Projects/pytorch/opt_min/'
experiment_no = '12210114'
# epochs = 5

data_len = 1000
test_len = 1000
batch_size = 100
log_loss = []

logging.basicConfig(level=logging.DEBUG,filename= path + experiment_no + '.log',
    filemode='w',format='[%(levelname)s:%(message)s]')
start = time.time()
train_set = utils.MyDataset_constraint_slsqp(n, m, mod, data_len)
data_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
print(time.time() - start)
test_set = utils.MyDataset_constraint_slsqp(n, m, mod, test_len)
test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=True)


net = utils.fc(n*n + n*m, 4096, 2048, 1024, n*m).cuda()
# net = utils.fc(n*n + n*m, 2048, 2048, 1024, n*m).cuda()

l2loss = utils.L2NomrLoss()
optimizer = optim.Adam(net.parameters(), lr= 1e-5)
# optimizer = optim.LBFGS(net.parameters(), lr = 0.01)
# optimizer = optim.SGD(net.parameters(), lr=1e-5, momentum=0.9)


start = time.time()
for epoch in range(epochs):  # loop over the dataset multiple times-
    utils.adjust_learning_rate(optimizer,epoch)
    train_set = utils.MyDataset_constraint_slsqp(n, m, mod, data_len)
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
            norm = torch.norm(outputs[j])
            if norm > mod:
                x = (outputs[j]/norm*mod).reshape((n,m)).float()
            else:
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


torch.save(net, path + experiment_no + 'net_params')


from matplotlib import pyplot as plt 
log_loss = [i for i in log_loss if i < 5 ]
x_axis = np.linspace(0,len(log_loss),len(log_loss))
plt.figure(figsize=(20,15))
plt.plot(x_axis,log_loss,label='loss')
plt.xlabel('epoch')
plt.legend()
# plt.show()
plt.savefig(path + experiment_no + '.png' )


avg_max = []

log_difference= []

for i, datas in enumerate(test_loader, 0):
    cofficient, label = datas
    a = cofficient[0].float()
    b = cofficient[1].float()
    outputs = net(cofficient[2].float()).reshape((batch_size, n, m))
    out = []
    for j in range(len(outputs)):
        norm = torch.norm(outputs[j])
        if norm > mod:
            x = outputs[j]/norm*mod
        else:
            x = outputs[j]
        out.append(utils.f_torch(x, a[j], b[j]))
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