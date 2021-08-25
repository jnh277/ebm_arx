import torch
import matplotlib.pyplot as plt
import Models
import pandas as pd
import scipy.stats as stats
import argparse
import scipy.linalg as linalg
import numpy as np


noise_form = 'gaussian'
scale = 2.0

network = Models.ScalarNet(hidden_dim=50)

network.load_state_dict(torch.load('results/scalar_example/network_'+noise_form+'.pt'))
data = pd.read_csv('results/scalar_example/data_'+noise_form+'.csv')

# calculate the least-squares solution


## plot the estimated distribution
x0 = 0.5/scale
x_test = x0*torch.ones((100,1))
y_test = torch.linspace(-1.,1.5,100).unsqueeze(1) / scale


scores = network(x_test,y_test)
dt = y_test[1]-y_test[0]
denom = scale*dt * scores.exp().sum().detach()

phat = scores.exp().detach()/denom

sig_m = 0.2
p_true = stats.norm(0.5*0.95, sig_m).pdf(y_test.detach()*scale)
plt.ylim([0,2.4])

plt.plot(y_test.detach()*scale,p_true,linewidth=4)
# plt.fill_between(scale*y_test.squeeze().detach(),p_true.squeeze(),p_true.squeeze()*0,color='blue',alpha=0.3)
plt.plot(y_test.detach()*scale,phat,linewidth=4,ls='--')
plt.xlabel('$y_t$',fontsize=20)
plt.ylabel('$p(y_t | y_{t-1}=0.5)$',fontsize=20)
plt.legend(['true distribution','learned distribution'])
plt.show()


# multi step ahead by numerical integration
K = 6
phats = torch.zeros((len(y_test),K+1))
phats[:, [0]] = phat

k = 1
x_bar, y_bar = torch.meshgrid((y_test.squeeze(), y_test.squeeze()))

for k in range(K):
    scores = network(x_bar.reshape(-1,1),y_bar.reshape(-1,1))
    scores_bar = scores.reshape(y_bar.size())
    denom = scale*dt * scores_bar.exp().sum(1).detach().reshape(-1,1)
    tmp = (scores_bar.detach().exp() * denom * phats[:,[k]]).sum(0)
    phat_new = tmp / tmp.sum() / scale / dt
    phats[:, k+1] = phat_new

for k in range(K):
    plt.subplot(2,3,k+1)
    plt.plot(y_test.detach()*scale,phats[:,k+1],linewidth=1)
    plt.xlabel('$y_{t+'+str(k+1)+'}$',fontsize=14)
    plt.ylabel('$p(y_{t+'+str(k+1)+'} | y_{t-1}=0.5)$',fontsize=14)

plt.tight_layout()
plt.show()



