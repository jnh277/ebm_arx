import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm
from chen_arx_example import GenerateChenData
import scipy.linalg as linalg
import Models
from Models import FullyConnectedNet



def evaluate(mdl, X_train, z_train, X_test, z_test):
    # One-step-ahead prediction
    z_pred_train = mdl.predict(X_train)
    z_pred_test = mdl.predict(X_test)
    return mse(z_train, z_pred_train), mse(z_test, z_pred_test), z_pred_train, z_pred_test


def mse(y_true, y_mdl):
    return np.mean((y_true - y_mdl)**2)


# ---- Main script ----
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    N = 250
    N_test = 500
    hidden_dim = 100
    batch_size = 128
    learning_rate = 0.001
    num_samples = 512
    num_epochs = 100
    stds = torch.zeros((1, 3))
    stds[0, 0] = 0.2
    stds[0, 1] = 0.4
    stds[0, 2] = 1.0
    noise_form = 'gaussian'
    save_results = True
    path = 'fc.pt'

    torch.manual_seed(117)
    np.random.seed(117)

    dataGen = GenerateChenData(noise_form=noise_form, sd_v=0.3, sd_w=0.3)
    X, Y, _, _ = dataGen(N, 1)
    X_test, Y_test, _, _ = dataGen(N_test, 1)

    scale = Y.max(0)
    X = X/scale
    Y = Y/scale
    X_test = X_test/scale
    Y_test = Y_test/scale

    net = FullyConnectedNet(n_hidden=150,n_interm_layers=4)
    net = net.fit(X, Y)

    net_ebm = Models.EBM_ARX_net(use_double=False,feature_net_dim=100,predictor_net_dim=100, decay_rate=0.99, num_epochs=600)
    net_ebm.fit(X, Y)

    Y_pred = net.predict(X)
    Y_pred_test = net.predict(X_test)
    pytorch_total_params = sum(p.numel() for p in net.net.parameters())
    print('FCC trainable parameters: ',pytorch_total_params)
    print('FCN Train rmse: ',np.sqrt(mse(Y*scale, Y_pred*scale)),'\nFCN Test rmse: ', np.sqrt(mse(Y_test*scale, Y_pred_test*scale)))

    # train EBM ARX Net

    yhat, prediction_scores = net_ebm.predict(X)
    yhat_test, prediction_scores = net_ebm.predict(X_test)
    pytorch_total_params = sum(p.numel() for p in net_ebm.net.parameters())
    print('EBM net trainable parameters: ',pytorch_total_params)
    print('EBM Train rmse: ',np.sqrt(mse(Y*scale, yhat.squeeze().numpy()*scale)),'\nEBM Test rmse: ', np.sqrt(mse(Y_test*scale, yhat_test.squeeze().numpy()*scale)))


    # compute linear baseline
    estim_param, _resid, _rank, _s = linalg.lstsq(X, Y)
    rmse_train_baseline = np.sqrt(mse(X @ estim_param * scale, Y * scale))
    rmse_test_baseline = np.sqrt(mse(X_test @ estim_param*scale,Y_test*scale))

    print('LSQR Train RMSE: ', rmse_train_baseline,'\nLSQR Test RMSE: ', rmse_test_baseline)

    fig, ax = plt.subplots()
    k = range(len(Y_test))
    ax.plot(k, Y_test, color='green', label='observed')
    ax.plot(k, Y_pred_test, color='blue',ls='--', label='fcn')
    ax.plot(k, yhat_test, ls='-.', label='ebm')
    ax.set_xlabel('k')
    ax.set_ylabel('y')
    ax.legend()
    plt.show()

