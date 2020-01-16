import numpy as np
import torch
from torch.utils import data
import torch.nn.init as init
import sys
sys.path.append('/media/veetsin/qwerty/Projects/pytorch/opt_min')
import utils

n = 50
m = 1
training_epochs = [10,100,1000,2000,5000,8000,10000,20000,40000]
training_epochs_to = [10, 20, 100, 200, 400,800,1600,4000,5000]
# training_epochs = [2, 3]

test_len = 1000
batch_size = 100

	
#generate data set
test_set = utils.MyDataset(n, m, test_len)
test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

import torch.optim as optim
from torch import nn

net = torch.load('/media/veetsin/qwerty/Projects/pytorch/opt_min/1216007net_params')  


x = torch.ones((n,m), requires_grad=True)

time_naive = []
time_nn = []
time_to = []

difference_nn = []
avg_time_naive = []
avg_time_nn = []
avg_time_to = []

avg_difference_naive = []
avg_difference_nn = []
avg_difference_to = []
import time 
for i in range(len(training_epochs)):
    # difference_to = []
    # for j, datas in enumerate(test_loader, 0):
    #     cofficient, label = datas
    #     a = cofficient[0]
    #     b = cofficient[1]
                
    #     start_time_to = time.time()
    #     outputs = net(cofficient[2].float()).reshape((100, 50, 1))
    #     outputs = outputs.detach()
    #     outputs = outputs.to('cpu')
    #     outputs.requires_grad = True
    #     optimizer_naive = torch.optim.Adam([outputs],lr = 1e-3)
    #     outputs.cuda()
    #     for step in range(training_epochs_to[i]):
    #         def closure():
    #             fx = []
    #             for l in range(len(x)):
    #                 pred = utils.f_torch(outputs[l].cuda().float(),a[l].float(),b[l].float())  
    #                 fx.append(pred)
    #                 fx = torch.stack(fx).float()
    #                 print(fx)
    #                 optimizer_naive.zero_grad()
    #                 fx.backward()
    #                 return fx
    #         out = optimizer_naive.step(closure)
    # difference_to.append( np.abs(out.to('cpu').detach().double() - label.to('cpu') ).numpy())
    # end_time_to = time.time()

    # avg_difference_to.append(np.mean(difference_to))
    # avg_time_to.append(end_time_to - start_time_to)  

    



    # if i == 0:
    #     start_time_nn = time.time()
    #     for j, datas in enumerate(test_loader, 0):
    #         cofficient, label = datas
    #         a = cofficient[0].float()
    #         b = cofficient[1].float()
    #         # outputs = net(cofficient[2].cuda().float()).reshape((10, 50, 1))
    #         outputs = net(cofficient[2].float()).reshape((100, 50, 1))
    #         out = []
    #         for l in range(len(outputs)):
    #             out.append(utils.f_torch(outputs[l],a[l],b[l]))
    #         out = torch.stack(out).double()
    #         difference = torch.mean( torch.abs(( (out.double() - label)) ))
    #         print(difference)
    #         difference_nn.append(difference.to('cpu').detach().numpy())
    #     end_time_nn = time.time()
    #     avg_difference_nn = np.mean(difference_nn)
    #     avg_time_nn.append(end_time_nn - start_time_nn)

    # else:
    #     pass


    difference_naive = []
    for j, datas in enumerate(test_loader, 0):
        cofficient, label = datas
        a = cofficient[0]
        b = cofficient[1]
                
        start_time_naive = time.time()
        x = torch.ones((batch_size, n, m), requires_grad=True)
        optimizer_naive = torch.optim.Adam([x],lr = 1e-1)
        for step in range(training_epochs_to[i]):
            def closure():
                fx = []
                for l in range(len(x)):
                    pred = utils.f_torch(x[l].cuda().float(),a[l].float(),b[l].float())  
                    fx.append(pred)
                    fx = torch.stack(fx).float()
#                    print(fx)
                    optimizer_naive.zero_grad()
                    fx.backward()
                    return fx
            out = optimizer_naive.step(closure)
    difference_naive.append( np.abs(out.to('cpu').detach().double() - label.to('cpu') ).numpy())
    end_time_naive = time.time()

    avg_difference_naive.append(np.mean(difference_naive))
    avg_time_naive.append(end_time_naive - start_time_naive)  






print(' average difference of naive optimizer, nn and transfer optimizing:')
print(avg_difference_naive, '\n', avg_difference_nn, '\n', avg_difference_to)


print(' average time of naive optimizer, nn and transfer optimizing:')
print(avg_time_naive, '\n', avg_time_nn, '\n', avg_time_to)


# roughly trained result net1216005

#  average difference of naive optimizer, nn and transfer optimizing:
# [715.843, 651.002, 500.415, 590.569, 510.797, 43.310, 0.697, 0.305]

#  0.6205657010281747

#  [0.996, 1.091, 0.638, 1.171, 0.365, 0.224, 0.234, 0.225]

# training_epochs_to = [2, 4, 10, 20, 100,1000,2000,4000]



#  average time of naive optimizer, nn and transfer optimizing:
# [0.002, 0.005, 0.014, 0.041, 0.116, 1.372, 2.779, 5.427]

#  [0.12960267066955566]

#  [0.006, 0.009, 0.018, 0.035, 0.127, 1.260, 2.701, 5.374]



# well trained result net 1216007

#  average difference of naive optimizer, nn and transfer optimizing:
#  average difference of naive optimizer, nn and transfer optimizing:
# [585.198748286388, 553.1502742204375, 527.3932084611438, 491.8597406958889, 287.66744229294767, 96.75686596628209, 3.8329080772006456, 0.24377953573998767, 0.26967902542185707]
#  0.43051556911608974
#  [0.4235851886573426, 0.3814776040410522, 0.5540171126350603, 0.2281113701345002, 0.2355113556623881, 0.29889888106298385, 0.516551980345097, 0.26031166019326785, 0.4100250039296005]

# training_epochs_to = [10, 20, 100, 200, 400,800,1600,4000,5000]

#  average time of naive optimizer, nn and transfer optimizing:
# [0.01337885856628418, 0.025032520294189453, 0.12251973152160645, 0.2351548671722412, 0.59230637550354, 1.1183099746704102, 2.4053702354431152, 4.902559280395508, 6.334798336029053]
#  [0.11944985389709473]
#  [0.0168759822845459, 0.02636551856994629, 0.1454486846923828, 0.3085649013519287, 0.4840047359466553, 1.2529327869415283, 2.008178949356079, 5.077316999435425, 6.304870128631592]


#one more in naive optimizer to see if optimize too much we will get worse result 
training_epochs = [10,100,1000,2000,5000,8000,10000,20000]
average difference of naive optimizer, nn and transfer optimizing:
[26.330011947087304, 42.12291444770641, 0.25000530803763854, 0.22447504046994077, 0.24890779904851748, 0.40014815708281903, 0.28623270862350997, 0.4070204338012749]
 []
 []
