import numpy as np
import torch
import matplotlib.pyplot as plt
import Models
import pickle5 as pickle
from coupledElectricDrives import DataScaler


with open('results/coupled_electric_drives/data.pkl','rb') as f:
    data = pickle.load(f)

with open('results/coupled_electric_drives/network.pkl','rb') as f:
    net = pickle.load(f)

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
pdfData = net.pdf_predict(phi_val)

# unscale outputs
yhat = yDS.unscale_data(yhat)

res = ()
for r in pdfData:
    res = res + (yDS.unscale_data(r),)
pdf, cdf, u95, l95, u99, l99, u65, l65, xt = res

plt.plot(yDS.unscale_data(yVal))
plt.plot(yDS.unscale_data(yhat.detach().numpy()))
plt.legend(['Meausrements', 'Predictions'])
plt.title('Test set predictions')
plt.xlabel('t')
plt.ylabel('y')
plt.show()

plt.plot(yVal, color='red', ls='None', marker='*')
plt.plot(yhat, color='blue')
plt.fill_between(np.arange(len(yVal)), u99, l99, alpha=0.1, color='b')
plt.fill_between(np.arange(len(yVal)), u95, l95, alpha=0.1, color='b')
plt.fill_between(np.arange(len(yVal)), u65, l65, alpha=0.1, color='b')
plt.xlabel('t', fontsize=20)
plt.ylabel('y', fontsize=20)
plt.xlim([50, 60])
plt.legend(['measured', 'mean pred', 'predicted $p(Y_t=y_t | X_t = x_t)$'])
plt.show()

plt.plot(xt, pdf[57])
plt.axvline(yVal[57], ls='--', color='k')
plt.xlabel('$y_{57}$', fontsize=20)
plt.ylabel('$p(y_{57}|X_{57}$)', fontsize=20)
plt.legend(['Predicted distribution', 'Measurement'], fontsize=14)
plt.show()

