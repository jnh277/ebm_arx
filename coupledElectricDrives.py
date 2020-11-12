import pandas as pd
import numpy as np
import torch
import pickle5 as pickle
import matplotlib.pyplot as plt
import Models
import scipy.linalg as linalg
from sklearn.model_selection import train_test_split
from Models import FullyConnectedNet

def build_phi_matrix(obs,order,inputs):
    "Builds the regressor matrix"
    no_obs = len(obs)
    max_delay = np.max((order[0],order[1]-1))
    phi = np.zeros((no_obs-max_delay, np.sum(order)))
    for i in range(order[0]):
        phi[:,i] = obs[max_delay-i-1:-i-1]
    for i in range(order[1]):
        phi[:,i+order[0]] = inputs[max_delay-i:no_obs-i]
    return phi


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")      # use gpu if available

class DataScaler(object):
    def __init__(self, data, upper: float = 1.0, lower: float = -1.0):
        self.upper = upper
        self.lower = lower
        self.data_max = data.max()
        self.data_min = data.min()
        self.data = data
        self.scaled_data = (data - self.data_min) / (self.data_max - self.data_min) * 2 - 1.0

    def scale_data(self,data):
        # data_scaled = ()
        # for d in data:
        data_scaled = (data - self.data_min) / (self.data_max - self.data_min) * 2 - 1.0
        return data_scaled

    def unscale_data(self,scaled_data):
        # data = ()
        # for sd in scaled_data:
        data =  (scaled_data + 1.0) / 2 * (self.data_max - self.data_min) + self.data_min
        return data


# ---- Main script ----
if __name__ == "__main__":
    save_results = False

    CEDdata = pd.read_csv('./data/coupledElectricDrives/DATAPRBS.csv')

    u = np.reshape(CEDdata[['u1', 'u2', 'u3']].to_numpy().T,(-1,))
    y = np.reshape(CEDdata[['z1', 'z2', 'z3']].to_numpy().T, (-1,))


    # CEDdata = pd.read_csv('./data/coupledElectricDrives/DATAUNIF.csv')
    #
    # u = np.reshape(CEDdata[['u11','u12']].to_numpy().T,(-1,))
    # y = np.reshape(CEDdata[['z11', 'z12']].to_numpy().T, (-1,))

    yDS = DataScaler(y)
    uDS = DataScaler(u)
    y = yDS.scaled_data
    u = uDS.scaled_data

    order = [3, 3]
    max_delay = np.max((order[0],order[1]-1))
    phi = build_phi_matrix(y, order, u)
    y = y[max_delay:]

    ## split randomly
    phi_est,phi_val,yEst,yVal = train_test_split(phi,y,train_size=750,random_state=52)

    N = len(yEst)
    N_test = len(yVal)

    net = Models.EBM_ARX_net(use_double=False,weight_decay=0.00,feature_net_dim=75,predictor_net_dim=75, decay_rate=0.99, num_epochs=600)
    net.fit(phi_est, yEst)
    training_losses = net.training_losses

    net_fcn = FullyConnectedNet(n_hidden=150, n_interm_layers=4)
    net_fcn.fit(phi_est, yEst)
    yhat_fcn = net_fcn.predict(phi_val)


    plt.plot(training_losses)
    plt.title('Training loss')
    plt.xlabel('epoch')
    plt.show()


    # make baseline predictions of test data set using least squares
    estim_param, _resid, _rank, _s = linalg.lstsq(phi_est, yEst)
    yhat_lsq= phi_val @ estim_param


    # # make predictions of test data set using trained EBM NN
    yhat, prediction_scores = net.predict(phi_val)
    pdfData = net.pdf_predict(phi_val)

    # unscale outputs
    yhat = yDS.unscale_data(yhat)
    yVal = yDS.unscale_data(yVal)
    yhat_lsq = yDS.unscale_data(yhat_lsq)

    res = ()
    for r in pdfData:
        res = res + (yDS.unscale_data(r),)
    pdf, cdf, u95, l95, u99, l99, u65, l65, xt = res

    plt.plot(prediction_scores[:100])
    plt.title('score during prediction stage')
    plt.show()
    #
    #
    plt.plot(yDS.unscale_data(yVal))
    plt.plot(yDS.unscale_data(yhat.detach().numpy()))
    plt.legend(['Meausrements','Predictions'])
    plt.title('Test set predictions')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.show()

    # pdf, cdf, u95, l95, u99, l99, u65, l65, xt = net.pdf_predict(phi_val)



    plt.plot(yVal, color='red', ls='None', marker='*')
    plt.plot(yhat, color='blue')
    plt.fill_between(np.arange(len(yVal)), u99, l99, alpha=0.1, color='b')
    plt.fill_between(np.arange(len(yVal)), u95, l95, alpha=0.1, color='b')
    plt.fill_between(np.arange(len(yVal)), u65, l65, alpha=0.1, color='b')
    plt.xlabel('t', fontsize=20)
    plt.ylabel('y', fontsize=20)
    plt.xlim([50, 60])
    plt.legend(['measured','mean pred', 'predicted $p(Y_t=y_t | X_t = x_t)$'])
    plt.show()

    plt.plot(xt,pdf[57])
    plt.axvline(yVal[57],ls='--',color='k')
    plt.xlabel('$y_{57}$',fontsize=20)
    plt.ylabel('$p(y_{57}|X_{57}$)',fontsize=20)
    plt.legend(['Predicted distribution','Measurement'],fontsize=14)
    plt.show()

    rmse_baseline = np.sqrt(np.mean((yhat_lsq - yVal) ** 2))
    e = yhat.squeeze() - yVal
    e_lsq = yhat_lsq - yVal
    #
    plt.plot(abs(e.detach()))
    plt.plot(abs(e_lsq))
    plt.ylabel('error magnitudes')
    plt.legend(['EBM','LSQ'])
    plt.show()

    # ind = abs(e) < 4*e.std()
    pytorch_total_params = sum(p.numel() for p in net.net.parameters())
    print('Total trainable parameters:',pytorch_total_params)
    rmse = torch.mean((e)**2).sqrt()
    print('Test RMSE')
    print('Least squares', rmse_baseline)
    print('EBM NN:', rmse.item())

    if save_results:
        data = {'phi_est':phi_est,
                'yEst':yEst,
                'phi_val':phi_val,
                'yVal':yVal,
                'order':order}
        with open('results/coupled_electric_drives/data.pkl',"wb") as f:
            pickle.dump(data, f)
        with open('results/coupled_electric_drives/network.pkl',"wb") as f:
            pickle.dump(net, f)
        with open('results/coupled_electric_drives/yDS.pkl', "wb") as f:
            pickle.dump(yDS, f)
        with open('results/coupled_electric_drives/uDS.pkl', "wb") as f:
            pickle.dump(uDS, f)
        with open('results/coupled_electric_drives/fcn.pkl',"wb") as f:
            pickle.dump(net_fcn, f)