import numpy as np
import torch
import matplotlib.pyplot as plt
import Models
import scipy.linalg as linalg
import scipy.stats as stats
import pickle5 as pickle


noise_model = 'mixture'     # alternatively, gaussian

if noise_model == 'mixture':
    with open('results/arx_example/data2.pkl','rb') as f:
        data = pickle.load(f)

    with open('results/arx_example/network2.pkl','rb') as f:
        net = pickle.load(f)
elif noise_model == 'gaussian':
    with open('results/arx_example/data.pkl', 'rb') as f:
        data = pickle.load(f)

    with open('results/arx_example/network.pkl', 'rb') as f:
        net = pickle.load(f)


scale = data['scale']
X = data['X']
Y = data['Y']
X_test = data['X_test']
Y_test = data['Y_test']

pdf, cdf, u95, l95, u99, l99, u65, l65, xt = net.pdf_predict(X_test)
xt = xt*scale
yhat, _ = net.predict(X_test)

mu = scale * torch.from_numpy(X_test).mm(torch.tensor([-0.7, 1.5, 0.5, 1.0],dtype=torch.double).unsqueeze(1))


# calculate the least-squares solution
estim_param, _resid, _rank, _s = linalg.lstsq(data['X'], np.expand_dims(data['Y'],1))
yhat_lsq = X_test @ estim_param
e_lsq = yhat_lsq - Y_test
std_lsq = e_lsq.std()

# work out the confidence region
pe_true = 0.4*stats.norm(0, 0.3).pdf(xt) + 0.6 * stats.norm(mu, 0.1).pdf(xt)
dx = (xt[1]-xt[0]).numpy()
cdf = np.cumsum(pe_true*dx)
ind = np.argmin(np.abs(cdf-0.025))
CI_t = -xt[ind].numpy()

plt.plot(scale * Y_test, color='red', ls='None', marker='*', label='Measured')
plt.plot(scale * yhat, color='blue', label='EB-NARX MAP')
plt.fill_between(np.arange(len(Y_test)),scale*u99,scale*l99, alpha=0.1, color='b', label=r'EB-NARX $p_{\theta}(y_t | x_t)$')
plt.fill_between(np.arange(len(Y_test)),scale*u95,scale*l95, alpha=0.1, color='b')
plt.fill_between(np.arange(len(Y_test)),scale*u65,scale*l65, alpha=0.1, color='b')
plt.plot(mu, color='orange', ls='-.', label='True mean')
if noise_model=='gaussian':
    plt.plot(mu+2*0.3, color='orange', ls='--', label='True +/- 2$\sigma$')
    plt.plot(mu-2*0.3, color='orange', ls='--')
elif noise_model=='mixture':
    plt.plot(mu+CI_t, color='orange', ls='--', label='True 95\% CI')
    plt.plot(mu-CI_t, color='orange', ls='--')
plt.xlabel('t', fontsize=20)
plt.ylabel('y', fontsize=20)
plt.xlim([50, 60])
plt.legend()
plt.show()

# ind = 56
ind = 56
mu = scale * (torch.tensor([-0.7, 1.5, 0.5, 1.0]) * X_test[ind, :]).sum()
if noise_model == 'gaussian':
    p_true = stats.norm(mu, 0.3).pdf(xt)
elif noise_model == 'mixture':
    p_true = 0.4*stats.norm(mu, 0.3).pdf(xt) + 0.6 * stats.norm(mu, 0.1).pdf(xt)

p_lsq = stats.norm(yhat_lsq[ind], std_lsq).pdf(xt)

plt.plot(xt, p_true, linewidth=3)
plt.fill_between(xt, p_true, 0 * p_true, alpha=0.3)
plt.plot(xt, pdf[ind]/scale, linewidth=3, ls='--')
plt.plot(xt, p_lsq, linewidth=3, ls='-.')
plt.axvline(scale*Y_test[ind], ls='--', color='k', linewidth=3)
plt.xlabel('$y_{56}$', fontsize=20)
plt.ylabel(r'$p_\theta(y_{56}|x_{56})$', fontsize=20)
plt.legend(['True', 'Estimated', 'Least-squares', 'Measurement'])
plt.show()

