import numpy as np
import torch
import matplotlib.pyplot as plt
import Models
import pickle5 as pickle
from coupledElectricDrives import DataScaler
import scipy.stats as stats


with open('results/coupled_electric_drives/data.pkl','rb') as f:
    data = pickle.load(f)

with open('results/coupled_electric_drives/network.pkl','rb') as f:
    net = pickle.load(f)

with open('results/coupled_electric_drives/fcn.pkl','rb') as f:
    net_fcn = pickle.load(f)

with open('results/coupled_electric_drives/uDS.pkl','rb') as f:
    uDS = pickle.load(f)

with open('results/coupled_electric_drives/yDS.pkl','rb') as f:
    yDS = pickle.load(f)

phi_est = data['phi_est']   # this is still scaled for easy passing into net
phi_val = data['phi_val']   # this is still scaled for easy passing into net
yEst = data['yEst']         # this has been unscaled
yVal = data['yVal']         # this has been unscaled

# # make predictions of test data set using trained EBM NN
yhat, prediction_scores = net.predict(phi_val)
pdf,cdf,u95, l95, u99, l99, u65, l65, xt = net.pdf_predict(phi_val)

## predict using FCN
yhat_fcn = net_fcn.predict(phi_val)

# unscale outputs
yhat = yDS.unscale_data(yhat)
yhat_fcn = yDS.unscale_data(yhat_fcn)

out = (u95, l95, u99, l99, u65, l65, xt)
res = ()
for r in out:
    res = res + (yDS.unscale_data(r),)
u95, l95, u99, l99, u65, l65, xt = res

## FCN error and sample variance
e_fcn = yhat_fcn - yVal
fcn_std = e_fcn.std()


plt.plot(yDS.unscale_data(yVal))
plt.plot(yDS.unscale_data(yhat.detach().numpy()))
plt.legend(['Meausrements', 'Predictions'])
plt.title('Test set predictions')
plt.xlabel('t')
plt.ylabel('y')
plt.show()


plt.plot(yVal, color='red', ls='None', marker='*', label='measured')
plt.plot(yhat, color='blue',label='EB-NARX MAP')
plt.plot(yhat_fcn, color='orange', ls='-.', label='FCN mean')
plt.fill_between(np.arange(len(yVal)), u99, l99, alpha=0.1, color='b', label=r'EB-NARX $p_\theta(y_t | x_t)$')
plt.fill_between(np.arange(len(yVal)), u95, l95, alpha=0.1, color='b')
plt.fill_between(np.arange(len(yVal)), u65, l65, alpha=0.1, color='b')
plt.plot(yhat_fcn+fcn_std*2, color='orange', ls='--', label='FCN +/- 2 stds')
plt.plot(yhat_fcn-fcn_std*2, color='orange', ls='--')
plt.xlabel('t', fontsize=20)
plt.ylabel('y', fontsize=20)
plt.xlim([50, 60])
plt.legend()
# plt.legend(['measured', 'MAP', 'predicted $p(Y_t=y_t | X_t = x_t)$', 'FCN +/- 2stds'])
plt.show()



ind = 40 # this one is unimodal but very non gaussian
p_fcn = stats.norm(yhat_fcn[ind], fcn_std).pdf(xt)
dt = xt[1] - xt[0]
plt.plot(xt, pdf[ind] / (pdf[ind]*dt).sum())
plt.plot(xt, p_fcn, ls='--')
plt.axvline(yVal[ind], ls='--', color='k')
plt.xlabel('$y_{'+str(ind)+'}$', fontsize=20)
plt.ylabel(r'$p_\theta(y_{'+str(ind)+'}|x_{'+str(ind)+'}$)', fontsize=20)
plt.legend(['EB-NARX', 'FCN', 'Measurement'], fontsize=14)
plt.show()

ind = 57 # this one is bimodal
p_fcn = stats.norm(yhat_fcn[ind], fcn_std).pdf(xt)
dt = xt[1] - xt[0]
plt.plot(xt, pdf[ind] / (pdf[ind]*dt).sum())
plt.plot(xt, p_fcn, ls='--')
plt.axvline(yVal[ind], ls='--', color='k')
plt.xlabel('$y_{'+str(ind)+'}$', fontsize=20)
plt.ylabel(r'$p(y_{'+str(ind)+'}|x_{'+str(ind)+'}$)', fontsize=20)
plt.legend(['EB-NARX', 'FCN', 'Measurement'], fontsize=14)
plt.show()

ind = 60 # non-gaussian uni modal
p_fcn = stats.norm(yhat_fcn[ind], fcn_std).pdf(xt)
dt = xt[1] - xt[0]
plt.plot(xt, pdf[ind] / (pdf[ind]*dt).sum())
plt.plot(xt, p_fcn, ls='--')
plt.axvline(yVal[ind], ls='--', color='k')
plt.xlabel('$y_{'+str(ind)+'}$', fontsize=20)
plt.ylabel(r'$p(y_{'+str(ind)+'}|x_{'+str(ind)+'}$)', fontsize=20)
plt.legend(['EB-NARX', 'FCN', 'Measurement'], fontsize=14)
plt.show()

