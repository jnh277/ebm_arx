import torch
import matplotlib.pyplot as plt
import Models
import pandas as pd
import scipy.stats as stats
import argparse
import scipy.linalg as linalg
import numpy as np

parser = argparse.ArgumentParser(description='Estimate NARX model for different n features / n samples rate.')
parser.add_argument('-m', '--noise_model', default='gaussian',
                    help='noise model default=gaussian, other options are'
                         'bimodal, and cauchy')


args, unk = parser.parse_known_args()

noise_form = args.noise_model            # this can be 'gaussian', or 'bimodal', or 'cauchy'
scale = 2.0

network = Models.ScalarNet(hidden_dim=50)

network.load_state_dict(torch.load('results/scalar_example/network_'+noise_form+'.pt'))
data = pd.read_csv('results/scalar_example/data_'+noise_form+'.csv')

# calculate the least-squares solution
estim_param, _resid, _rank, _s = linalg.lstsq(np.expand_dims(data['X'],1), np.expand_dims(data['Y'],1))
e_lsq = np.expand_dims(data['X'],1) @ estim_param - np.expand_dims(data['Y'],1)
mu_lsq = e_lsq.mean()
std_lsq = e_lsq.std()
print('Least-squares std: ',std_lsq)

## plot the estimated distribution
x0 = 0.0
x_test = x0*torch.ones((500,1))
y_test = torch.linspace(-0.5,0.5,500).unsqueeze(1)


# lsq distribution
p_lsq = stats.norm(mu_lsq * scale, std_lsq * scale).pdf(scale * y_test.detach())

scores = network(x_test,y_test)
dt = y_test[1]-y_test[0]
denom = scale*dt * scores.exp().sum().detach()

if noise_form == 'gaussian':
    sig_m = 0.2
    p_true = stats.norm(0, sig_m).pdf(scale * y_test.detach())
    plt.ylim([0,2.4])
elif noise_form == 'bimodal':
    sig_m = 0.1
    p_true = 0.5*stats.norm(0.4, sig_m).pdf(scale * y_test.detach())+0.5*stats.norm(-0.4, sig_m).pdf(scale * y_test.detach())
    plt.ylim([0,2.4])
elif noise_form == 'cauchy':
    sig_m = 0.2
    x_test = x0 * torch.ones((500, 1))
    y_test = torch.linspace(-1.5, 1.5, 500).unsqueeze(1)
    scores = network(x_test, y_test)
    dt = y_test[1] - y_test[0]
    denom = scale * dt * scores.exp().sum().detach()
    p_true = stats.cauchy(0.0, sig_m).pdf(scale * y_test.detach())
    plt.ylim([0,2.05])
plt.plot(scale*y_test.detach(),p_true,linewidth=4)
plt.fill_between(scale*y_test.squeeze().detach(),p_true.squeeze(),p_true.squeeze()*0,color='blue',alpha=0.3)
plt.plot(scale*y_test.detach(),scores.exp().detach()/denom,linewidth=4,ls='--')
plt.plot(scale * y_test.detach(),p_lsq.squeeze(),linewidth=4,ls='-.')
plt.xlabel('$e_t$',fontsize=20)
plt.ylabel('$p(e_t)$',fontsize=20)
# plt.title("noise distribution")
plt.legend(['True distribution','Learned distribution','Least-squares'])
plt.show()
