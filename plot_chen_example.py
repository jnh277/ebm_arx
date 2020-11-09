import numpy as np
import torch
import matplotlib.pyplot as plt
import Models
import scipy.stats as stats
import pickle5 as pickle


with open('results/chen_model/data.pkl','rb') as f:
    data = pickle.load(f)

network = Models.ARXnet(x_dim=4,y_dim=1,hidden_dim=data['hidden_dim'])
network.load_state_dict(torch.load('results/chen_model/network.pt'))

scale = data['scale']
X = torch.from_numpy(data['X'])
Y = torch.from_numpy(data['Y'])
X_test = torch.from_numpy(data['X_test'])
Y_test = torch.from_numpy(data['Y_test'])

yhat_init, yhat_samples, scores_samples = Models.init_predict(X_test.double(),
                                                              Y_test.clone().detach().double().unsqueeze(1),
                                                              network.double(), 2028, [-1.0, 1.0])

xt = scale * np.linspace(-1, 1, 2028)
dt = xt[1] - xt[0]
denom = scores_samples.exp().detach().sum(1) * dt
cdf = np.cumsum(scores_samples.exp().detach() / np.reshape(denom,(-1,1)) *dt,axis=1)
u95 = xt[np.argmin(abs(cdf-0.975), 1)]
l95 = xt[np.argmin(abs(cdf-0.025), 1)]
u99 = xt[np.argmin(abs(cdf-0.995), 1)]
l99 = xt[np.argmin(abs(cdf-0.005), 1)]
u65 = xt[np.argmin(abs(cdf-0.825), 1)]
l65 = xt[np.argmin(abs(cdf-0.175), 1)]

plt.plot(scale * Y_test.detach(), color='red', ls='None', marker='*')
plt.fill_between(np.arange(len(Y_test)),u99,l99,alpha=0.1,color='b')
plt.fill_between(np.arange(len(Y_test)),u95,l95,alpha=0.1,color='b')
plt.fill_between(np.arange(len(Y_test)),u65,l65,alpha=0.1,color='b')
plt.xlabel('t', fontsize=20)
plt.ylabel('y', fontsize=20)
plt.xlim([25, 55])
plt.legend(['measured', 'predicted $p(Y_t=y_t | X_t = x_t$)'])
plt.show()

# ind = 50
ind = 30
xt = scale * np.linspace(-1, 1, 2028)
# mu = scale * (torch.tensor([-0.7, 1.5, 0.5, 1.0]) * X_test[ind, :]).sum()
# p_true = stats.norm(mu, 0.3).pdf(xt)

dt = xt[1] - xt[0]
denom = scores_samples[ind].exp().detach().sum() * dt
plt.plot(xt, scores_samples[ind].exp().detach() / denom, linewidth=3)
plt.fill_between(xt, scores_samples[ind].exp().detach() / denom, 0 * xt, alpha=0.3)
plt.axvline(scale*Y_test[ind], ls='--', color='k', linewidth=3)
plt.xlabel('$y_{30}$', fontsize=20)
plt.ylabel('$p(Y_{30}=y_{30}|X_{30}=x_{30})$', fontsize=20)
plt.legend(['True', 'Estimated', 'measurement'])
plt.show()

ind = 50
xt = scale * np.linspace(-1, 1, 2028)
dt = xt[1] - xt[0]


denom = scores_samples[ind].exp().detach().sum() * dt
plt.plot(xt, scores_samples[ind].exp().detach() / denom, linewidth=3)
plt.fill_between(xt, scores_samples[ind].exp().detach() / denom, 0 * xt, alpha=0.3)
plt.axvline(scale*Y_test[ind], ls='--', color='k', linewidth=3)
plt.xlabel('$y_{50}$', fontsize=20)
plt.ylabel('$p(Y_{50}=y_{50}|X_{50}=x_{50})$', fontsize=20)
plt.legend(['Estimated', 'measurement'])
plt.show()
