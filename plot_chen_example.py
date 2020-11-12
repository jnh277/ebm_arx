import numpy as np
import torch
import matplotlib.pyplot as plt
import Models
import scipy.stats as stats
import pickle5 as pickle

with open('results/chen_model/data.pkl','rb') as f:
    data = pickle.load(f)

with open('results/chen_model/network.pkl','rb') as f:
    net = pickle.load(f)

with open('results/chen_model/fcn.pkl','rb') as f:
    net_fcn = pickle.load(f)

scale = data['scale']
X = data['X']
Y = data['Y']
X_test = data['X_test']
Y_test = data['Y_test']


pdf, cdf, u95, l95, u99, l99, u65, l65, xt = net.pdf_predict(X_test)
xt = xt*scale
yhat, _ = net.predict(X_test)

yhat_fcn = net_fcn.predict(X_test)

e_fcn = scale*Y_test - scale*yhat_fcn

fcn_std = e_fcn.std()

plt.plot(scale * Y_test, color='red', ls='None', marker='*')
plt.plot(scale * yhat, color='blue')
plt.fill_between(np.arange(len(Y_test)),scale*u99,scale*l99,alpha=0.1,color='b')
plt.fill_between(np.arange(len(Y_test)),scale*u95,scale*l95,alpha=0.1,color='b')
plt.fill_between(np.arange(len(Y_test)),scale*u65,scale*l65,alpha=0.1,color='b')
plt.xlabel('t', fontsize=20)
plt.ylabel('y', fontsize=20)
plt.xlim([35, 50])
plt.legend(['Measured', 'MAP', 'Predicted $p(Y_t=y_t | X_t = x_t)$'])
plt.show()


ind = 47

dt = xt[1] - xt[0]
p_fcn = stats.norm(yhat_fcn[ind]*scale, fcn_std).pdf(xt)
# p_true = p_true / (p_true * dt.numpy()).sum()

plt.plot(xt, pdf[ind]/scale, linewidth=3)
plt.fill_between(xt, pdf[ind]/scale, 0 * pdf[ind]/scale, alpha=0.3)
plt.plot(xt, p_fcn, linewidth=3, ls='--')
plt.axvline(scale*Y_test[ind], ls='--', color='k', linewidth=3)
plt.xlabel('$y_{'+str(ind)+'}$', fontsize=20)
plt.ylabel('$p(Y_{'+str(ind)+'}=y_{'+str(ind)+'}|X_{'+str(ind)+'}=x_{'+str(ind)+'})$', fontsize=20)
plt.legend(['EBM', 'FCN', 'Measurement'])
plt.show()