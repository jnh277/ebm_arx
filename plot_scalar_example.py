import torch
import matplotlib.pyplot as plt
import Models
import numpy as np
import pandas as pd
import scipy.stats as stats


noise_form = 'gaussian'            # this can be 'gaussian', or 'bimodal', or 'cauchy'
scale = 2.0

network = Models.ScalarNet(hidden_dim=50)

network.load_state_dict(torch.load('results/scalar_example/network_'+noise_form+'.pt'))
data = pd.read_csv('results/scalar_example/data_'+noise_form+'.csv')

## plot the estimated distribution
x0 = 0.0
x_test = x0*torch.ones((500,1))
y_test = torch.linspace(-0.5,0.5,500).unsqueeze(1)

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
plt.xlabel('$e_t$',fontsize=20)
plt.ylabel('$p(e_t)$',fontsize=20)
# plt.title("noise distribution")
plt.legend(['True distribution','Learned distribution'])
plt.xlim([-1,1])
plt.show()

if noise_form == 'gaussian':    # then plot the predicted
    Y = torch.tensor(data['Y'])
    X = torch.tensor(data['X'])
    yhat_init, yhat_samples, scores_samples = Models.init_predict(X[:49].unsqueeze(1).double(),
                                                                  Y[:49].clone().detach().double().unsqueeze(1),
                                                                  network.double(), 2028, [-1.0, 1.0])

    # ind = np.argmax(scores_samples.detach(),1)
    # yhat = yhat_

    xt = scale * np.linspace(-1, 1, 2028)
    dt = xt[1] - xt[0]
    denom = scores_samples.exp().detach().sum(1) * dt
    cdf = np.cumsum(scores_samples.exp().detach() / np.reshape(denom, (-1, 1)) * dt, axis=1)
    u95 = xt[np.argmin(abs(cdf - 0.975), 1)]
    l95 = xt[np.argmin(abs(cdf - 0.025), 1)]
    u99 = xt[np.argmin(abs(cdf - 0.995), 1)]
    l99 = xt[np.argmin(abs(cdf - 0.005), 1)]
    u65 = xt[np.argmin(abs(cdf - 0.825), 1)]
    l65 = xt[np.argmin(abs(cdf - 0.175), 1)]
    plt.fill_between(np.arange(len(Y[:49])), u99, l99, alpha=0.1, color='b')
    plt.fill_between(np.arange(len(Y[:49])), u95, l95, alpha=0.1, color='b')
    plt.fill_between(np.arange(len(Y[:49])), u65, l65, alpha=0.1, color='b')
    plt.plot(scale * Y[:49].detach(), color='red', ls='None', marker='*')
    plt.plot(scale * yhat_init[:49].detach(), color='blue')
    plt.xlabel('t', fontsize=20)
    plt.ylabel('y', fontsize=20)
    plt.xlim([15, 50])
    plt.legend(['Measured', 'MAP', 'Predicted $p(Y_t=y_t | X_t = x_t)$'])
    plt.show()