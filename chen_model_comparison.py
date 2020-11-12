import torch
import numpy as np
from chen_arx_example import GenerateChenData
import Models
from Models import FullyConnectedNet
import pickle5 as pickle
import random
import itertools
import pandas as pd
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
    N = 1000
    sigma = 0.3

    sd_w = sigma
    sd_v = sigma
    N_test = 500

    save_offset = 0

    num_exps = 500

    batch_size = 128
    learning_rate = 0.001
    num_samples = 512
    num_epochs = 100
    stds = torch.zeros((1, 3))
    stds[0, 0] = 0.2
    stds[0, 1] = 0.4
    stds[0, 2] = 1.0
    noise_form = 'gaussian'



    batch_sizes = [32, 64, 128]

    fcn_layers = [2, 3, 4]
    fcn_hidden_dims = [50, 75, 100, 150, 200, 300]
    fcn_acts = ['tanh', 'relu']

    fcn_options_list = list(itertools.product(fcn_layers,fcn_hidden_dims,
                                                         fcn_acts,
                                                         batch_sizes))
    random.shuffle(fcn_options_list)
    fcn_options = itertools.cycle(fcn_options_list)

    feature_net_dims = (50, 75, 100, 150, 200, 300)
    predictor_net_dims = (50, 75, 100, 150, 200, 300)

    ebm_options_list =list(itertools.product(feature_net_dims,predictor_net_dims,
                                                         batch_sizes))
    random.shuffle(ebm_options_list)
    ebm_options = itertools.cycle(ebm_options_list)

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

    with open('results/chen_comparison/sigma'+str(int(sigma*10))+'N'+str(N)+'/data.pkl', "wb") as f:
        pickle.dump(data, f)

    fcn_train_mse = []
    fcn_test_mse = []
    fcn_num_params = []
    ebm_train_mse = []
    ebm_test_mse = []
    ebm_num_params = []
    ebm_train_score = []
    ebm_test_score = []
    fcn_hidden_dims = []
    fcn_layers = []
    fcn_nonlinearity = []
    fcn_batch_size = []
    ebm_feature_dim = []
    ebm_predictor_dim = []
    ebm_batch_size = []

    for exp in range(num_exps):
        print('Running experiment ',exp)
        seed = np.random.randint(1,1000)

        fcn_opt = next(fcn_options)         # get next lot of FCN hyperparameters to try

        net = FullyConnectedNet(n_hidden=fcn_opt[1], decay_rate=0.99,
                                n_interm_layers=fcn_opt[0], random_state=seed,
                                nonlinearity=fcn_opt[2],epochs=500,
                                batch_size=fcn_opt[3])
        net = net.fit(X, Y)
        Y_pred = net.predict(X)
        Y_pred_test = net.predict(X_test)

        fcn_train_mse.append(mse(Y*scale, Y_pred*scale))
        fcn_test_mse.append(mse(Y_test*scale, Y_pred_test*scale))
        fcn_num_params.append(sum(p.numel() for p in net.net.parameters()))
        fcn_layers.append(fcn_opt[0])
        fcn_hidden_dims.append(fcn_opt[1])
        fcn_batch_size.append(fcn_opt[3])
        fcn_nonlinearity.append(fcn_opt[2])

        ebm_opt = next(ebm_options)     # get next lof of EBM hyperparameters to try

        net_ebm = Models.EBM_ARX_net(use_double=False,feature_net_dim=ebm_opt[0],
                                     predictor_net_dim=ebm_opt[1], decay_rate=0.99,
                                     num_epochs=500, random_state=seed,
                                     batch_size=ebm_opt[2])
        net_ebm.fit(X, Y)
        Y_pred_ebm, train_scores = net_ebm.predict(X)
        Y_pred_ebm_test, test_scores = net_ebm.predict(X_test)

        print('FCN # params: ', fcn_num_params[exp])
        print('FCN Train MSE: ',fcn_train_mse[exp])
        print('FCN Test MSE: ', fcn_test_mse[exp])

        ebm_train_mse.append(mse(Y*scale, Y_pred_ebm.squeeze().numpy()*scale))
        ebm_test_mse.append(mse(Y_test*scale, Y_pred_ebm_test.squeeze().numpy()*scale))
        ebm_num_params.append(sum(p.numel() for p in net_ebm.net.parameters()))
        ebm_train_score.append(train_scores[-1])
        ebm_test_score.append(test_scores[-1])
        ebm_feature_dim.append(ebm_opt[0])
        ebm_predictor_dim.append(ebm_opt[1])
        ebm_batch_size.append(ebm_opt[2])

        print('EBM # params: ', ebm_num_params[exp])
        print('EBM Train MSE: ', ebm_train_mse[exp])
        print('EBM Test MSE: ', ebm_test_mse[exp])
        print('\n')
        print('Best so far')
        print('FCN Test MSE: ',np.min(fcn_test_mse))
        print('EBM Test MSE: ',np.min(ebm_test_mse))

        with open('results/chen_comparison/sigma'+str(int(sigma*10))+'N'+str(N)+'/fcn'+str(exp)+'.pkl',"wb") as f:
            pickle.dump(net, f)

        with open('results/chen_comparison/sigma'+str(int(sigma*10))+'N'+str(N)+'/ebm' + str(exp) + '.pkl', "wb") as f:
            pickle.dump(net_ebm, f)

    print('Best results')
    print('FCN Test MSE: ', np.min(fcn_test_mse))
    print('EBM Test MSE: ', np.min(ebm_test_mse))


evaluations = {'fcn_train_mse':fcn_train_mse,
               'fcn_test_mse':fcn_test_mse,
               'fcn_num_params':fcn_num_params,
               'ebm_train_mse':ebm_train_mse,
               'ebm_test_mse':ebm_test_mse,
               'ebm_num_params':ebm_num_params,
               'ebm_test_score':ebm_test_score,
               'ebm_train_score':ebm_train_score,
               'fcn_layers':fcn_layers,
               'fcn_hidden_dims':fcn_hidden_dims,
               'fcn_nonlinearity':fcn_nonlinearity,
               'fcn_batch_size':fcn_batch_size,
               'ebm_feature_dim':ebm_feature_dim,
               'ebm_predictor_dim':ebm_predictor_dim,
               'ebm_batch_size':ebm_batch_size}

df = pd.DataFrame(evaluations)
df.to_csv('results/chen_comparison/sigma'+str(int(sigma*10))+'N'+str(N)+'/evals.csv')
# with open('results/chen_comparison/N250/ebm41.pkl','rb') as f:
#     net = pickle.load(f)


