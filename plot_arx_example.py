import numpy as np
import torch
import matplotlib.pyplot as plt
import Models
import scipy.stats as stats
import pickle5 as pickle


with open('results/arx_example/data.pkl','rb') as f:
    data = pickle.load(f)

network = Models.ARXnet(x_dim=4,y_dim=1,hidden_dim=data['hidden_dim'])
network.load_state_dict(torch.load('results/arx_example/network.pt'))

scale = data['scale']
X = torch.from_numpy(data['X'])
Y = torch.from_numpy(data['Y'])
X_test = torch.from_numpy(data['X_test'])
Y_test = torch.from_numpy(data['Y_test'])

yhat_init, yhat_samples, scores_samples = Models.init_predict(X_test.double(),
                                                              Y_test.clone().detach().double().unsqueeze(1),
                                                              network.double(), 2028, [-1.0, 1.0])

scores = scores_samples.detach().exp().numpy()
scores_max = np.max(scores, 1)
scores = scores / scores_max.reshape(-1, 1)
plt.contour(np.arange(0, len(Y_test) + 0), scale * np.linspace(-1, 1, 2028), scores.T, 30)
plt.plot(scale * Y_test.detach(), color='red', ls='None', marker='*')
plt.xlabel('t', fontsize=20)
plt.ylabel('y', fontsize=20)
plt.xlim([50, 60])
plt.legend(['measured', 'predicted $p(Y_t=y_t | X_t = x_t$'])
plt.show()

ind = 50
xt = scale * np.linspace(-1, 1, 2028)
mu = scale * (torch.tensor([-0.7, 1.5, 0.5, 1.0]) * X_test[ind, :]).sum()
p_true = stats.norm(mu, 0.3).pdf(xt)

dt = xt[1] - xt[0]
denom = scores_samples[ind].exp().detach().sum() * dt
plt.plot(xt, p_true, linewidth=3)
plt.fill_between(xt, p_true, 0 * p_true, alpha=0.3)
plt.plot(xt, scores_samples[ind].exp().detach() / denom, linewidth=3, ls='--')
plt.axvline(Y_test[ind], ls='--', color='k', linewidth=3)
plt.xlabel('$y_{50}$', fontsize=20)
plt.ylabel('$p(Y_{50}=y_{50}|X_{50}=x_{50})$', fontsize=20)
plt.legend(['True', 'Estimated', 'measurement'])
plt.show()

