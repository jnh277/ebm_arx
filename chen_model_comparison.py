import torch
import numpy as np
from chen_arx_example import GenerateChenData
import Models
from Models import FullyConnectedNet
import pickle5 as pickle
import random
import matplotlib.pyplot as plt


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
    N = 500
    sd_w = 0.3
    sd_v = 0.3
    N_test = 500

    num_exps = 100

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

    fcn_layers = (2, 3, 4, 5, 6)
    fcn_hidden_dims = (25, 50, 75, 100, 150, 200, 300)
    feature_net_dims = (25, 50, 75, 100, 150, 200, 300)
    predictor_net_dims = (25, 50, 75, 100, 150, 200, 300)

    torch.manual_seed(117)
    np.random.seed(117)

    dataGen = GenerateChenData(noise_form=noise_form, sd_v=sd_v, sd_w=sd_w)
    X, Y, _, _ = dataGen(N, 1)
    X_test, Y_test, _, _ = dataGen(N_test, 1)

    scale = Y.max(0)
    X = X/scale
    Y = Y/scale
    X_test = X_test/scale
    Y_test = Y_test/scale

    data = {"scale": scale,
            "sd_v": sd_v,
            "sd_w": sd_w,
            "X": X,
            "Y": Y,
            "X_test": X_test,
            "Y_test": Y_test}

    with open('results/chen_comparison/N'+str(N)+'/data.pkl', "wb") as f:
        pickle.dump(data, f)

    fcn_train_mse = []
    fcn_test_mse = []
    fcn_num_params = []
    ebm_train_mse = []
    ebm_test_mse = []
    ebm_num_params = []

    for exp in range(num_exps):
        print('Running experiment ',exp)
        seed = np.random.randint(1,1000)

        net = FullyConnectedNet(n_hidden=random.choice(fcn_hidden_dims), decay_rate=0.99,
                                n_interm_layers=random.choice(fcn_layers), random_state=seed,
                                epochs=500)
        net = net.fit(X, Y)
        Y_pred = net.predict(X)
        Y_pred_test = net.predict(X_test)

        fcn_train_mse.append(mse(Y*scale, Y_pred*scale))
        fcn_test_mse.append(mse(Y_test*scale, Y_pred_test*scale))
        fcn_num_params.append(sum(p.numel() for p in net.net.parameters()))

        net_ebm = Models.EBM_ARX_net(use_double=False,feature_net_dim=random.choice(feature_net_dims),
                                     predictor_net_dim=random.choice(predictor_net_dims), decay_rate=0.99,
                                     num_epochs=500, random_state=seed)
        net_ebm.fit(X, Y)
        Y_pred_ebm, prediction_scores = net_ebm.predict(X)
        Y_pred_ebm_test, prediction_scores = net_ebm.predict(X_test)

        print('FCN # params: ', fcn_num_params[exp])
        print('FCN Train MSE: ',fcn_train_mse[exp])
        print('FCN Test MSE: ', fcn_test_mse[exp])

        ebm_train_mse.append(mse(Y*scale, Y_pred_ebm.squeeze().numpy()*scale))
        ebm_test_mse.append(mse(Y_test*scale, Y_pred_ebm_test.squeeze().numpy()*scale))
        ebm_num_params.append(sum(p.numel() for p in net_ebm.net.parameters()))

        print('EBM # params: ', ebm_num_params[exp])
        print('EBM Train MSE: ', ebm_train_mse[exp])
        print('EBM Test MSE: ', ebm_test_mse[exp])

        with open('results/chen_comparison/N'+str(N)+'/fcn'+str(exp)+'.pkl',"wb") as f:
            pickle.dump(net, f)

        with open('results/chen_comparison/N'+str(N)+'/ebm' + str(exp) + '.pkl', "wb") as f:
            pickle.dump(net_ebm, f)

    print('Best results')
    print('FCN Test MSE: ', np.min(fcn_test_mse))
    print('EBM Test MSE: ', np.min(ebm_test_mse))


# with open('results/chen_comparison/N250/ebm41.pkl','rb') as f:
#     net = pickle.load(f)


