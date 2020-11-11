import numpy as np
import torch
import matplotlib.pyplot as plt
import Models
import scipy.stats as stats
import pickle5 as pickle

with open('results/chen_model/data.pkl','rb') as f:
    data = pickle.load(f)

with open('results/arx_example/network.pkl','rb') as f:
    net = pickle.load(f)

scale = data['scale']
X = data['X']
Y = data['Y']
X_test = data['X_test']
Y_test = data['Y_test']


pdf, cdf, u95, l95, u99, l99, u65, l65, xt = net.pdf_predict(X_test)
xt = xt*scale
yhat, _ = net.predict(X_test)


plt.plot(scale * Y_test, color='red', ls='None', marker='*')
plt.plot(scale * yhat, color='blue')
plt.fill_between(np.arange(len(Y_test)),scale*u99,scale*l99,alpha=0.1,color='b')
plt.fill_between(np.arange(len(Y_test)),scale*u95,scale*l95,alpha=0.1,color='b')
plt.fill_between(np.arange(len(Y_test)),scale*u65,scale*l65,alpha=0.1,color='b')
plt.xlabel('t', fontsize=20)
plt.ylabel('y', fontsize=20)
plt.xlim([50, 60])
plt.legend(['measured', 'mean', 'predicted $p(Y_t=y_t | X_t = x_t)$'])
plt.show()


ind = 53

plt.plot(xt, pdf[ind]/scale, linewidth=3)
plt.fill_between(xt, pdf[ind]/scale, 0 * pdf[ind]/scale, alpha=0.3)
plt.axvline(scale*Y_test[ind], ls='--', color='k', linewidth=3)
plt.xlabel('$y_{50}$', fontsize=20)
plt.ylabel('$p(Y_{50}=y_{50}|X_{50}=x_{50})$', fontsize=20)
plt.legend(['Estimated', 'measurement'])
plt.show()