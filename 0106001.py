#replace parameter to complex number:fail
#0106001 002 difference: lr 1 to 100 ;data 1000 to 2000; epoch 300 to 1000
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
p = 15 #power limit
epochs = 1000
path = '/media/veetsin/qwerty/Projects/pytorch/opt_min/'
experiment_no = '0106002'


data_len = 2000
test_len = 100
batch_size = 100
log_loss = []

def f_precoding_cuda(w, h, a):#funtion d
    diag = torch.diag(torch.mm(h, w.reshape(n, k)))
    sinr = [(diag[i])**2/(1 + sum(      torch.cat( [   (diag**2)[0:i],(diag**2)[i+1 :] ])        )  ) for i in range(k) ]
    return -sum([a[i]*torch.log2(1 + sinr[i]) for i in range(k) ])

logging.basicConfig(level=logging.DEBUG,filename= path + experiment_no + '.log',
    filemode='w',format='[%(levelname)s:%(message)s]')
start = time.time()
train_set = utils.MyDataset_precoding_targetx(k, n, p, data_len)
data_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
print(time.time() - start)
test_set = utils.MyDataset_precoding_targetx(k, n, p, test_len)
test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=True)


input_size = k*n + k
net = utils.fc_precoding(input_size, 60*input_size, 50*input_size, 40*input_size, 30*input_size, 20*input_size, 10*input_size, n*k).cuda()
# net = utils.fc(n*n + n*m, 2048, 2048, 1024, n*m).cuda()

# l2loss = utils.L2NomrLoss()
fnormloss = utils.FNormLoss()
optimizer = optim.Adam(net.parameters(), lr= 100)
# optimizer = optim.LBFGS(net.parameters(), lr = 0.01)
# optimizer = optim.SGD(net.parameters(), lr=1, momentum=0.9)


start = time.time()
for epoch in range(epochs):  # loop over the dataset multiple times-
    utils.adjust_learning_rate_precoding_targetx(optimizer,epoch)
    # train_set = utils.MyDataset_constraint(n, m, mod, data_len)
    # data_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    for i, datas in enumerate(data_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        # zero the parameter gradients
        # def closure():
        optimizer.zero_grad()
        cofficient, label = datas
        outputs = net(cofficient[2].float()) #20*600
        h_list = cofficient[0]
        a_list = cofficient[1]
    #     break
    # break
        w_list = []
        for j in range(batch_size):
            w = outputs[j].reshape((n,k)).float()
            norm_sum =   sum([np.linalg.norm((w.detach().to('cpu').numpy()).T[i]) for i in range(k)]) 
            if norm_sum > p:
                w = w*p/norm_sum
            else:
                pass
            w_list.append(w)
        w_list = torch.stack(w_list).float()

        label = label.reshape(batch_size, n, k).float()
        # print(fx)
        # print(label)

        loss = fnormloss(w_list, label)
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
    h = cofficient[0].float()
    a = cofficient[1].float()
    w = net(cofficient[2].float()).reshape((batch_size, n, k))
    out = []
    for j in range(len(w)):
        norm_sum =  sum([np.linalg.norm((w[j].detach().to('cpu').numpy()).T[i]) for i in range(k)]) 
        if norm_sum > p:
            wj = w[j]*p/norm_sum
        else:
            wj = w[j]
        # wj = w[j]
        out.append(wj)
    out = torch.stack(out).double()
    label = label.reshape(batch_size, n, k).double()
    loss = fnormloss(out, label)
    batch_max = max([torch.norm(out[i]-label[i] ) for i in range(batch_size)]).to('cpu').detach().numpy()
    avg_max.append(batch_max)

    log_difference.append(loss.detach().to('cpu').numpy())

  
    print('test batch: %d, difference: %.3f batch_max_difference: %.3f' %  (i + 1, loss.detach().to('cpu').numpy() , batch_max ))
    logging.info('test batch: %d, difference: %.3f batch_max_difference: %.3f' %  (i + 1, loss.detach().to('cpu').numpy() , batch_max ))



print('average test difference: %.3f, average max difference: %.3f, max difference: %.3f' % ( sum(log_difference)/len(log_difference), sum(avg_max)/len(avg_max), max(avg_max)))
logging.info('average test difference: %.3f, average max difference: %.3f, max difference: %.3f' % ( sum(log_difference)/len(log_difference), sum(avg_max)/len(avg_max), max(avg_max)))