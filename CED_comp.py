import torch
import numpy as np
import Models
from Models import FullyConnectedNet
import pickle5 as pickle
import random
import itertools
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


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

def mse(y_true, y_mdl):
    return np.mean((y_true - y_mdl)**2)

# ---- Main script ----
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    save_start = 0
    num_exps = 500

    learning_rate = 0.001
    num_samples = 512
    stds = torch.zeros((1, 3))
    stds[0, 0] = 0.2
    stds[0, 1] = 0.4
    stds[0, 2] = 1.0



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

    ## load data
    CEDdata = pd.read_csv('./data/coupledElectricDrives/DATAPRBS.csv')

    u = np.reshape(CEDdata[['u1', 'u2', 'u3']].to_numpy().T,(-1,))
    y = np.reshape(CEDdata[['z1', 'z2', 'z3']].to_numpy().T, (-1,))

    yDS = DataScaler(y)
    uDS = DataScaler(u)
    y = yDS.scaled_data
    u = uDS.scaled_data

    order = [3, 3]
    max_delay = np.max((order[0],order[1]-1))
    phi = build_phi_matrix(y, order, u)
    y = y[max_delay:]

    ## split randomly
    X, X_test, Y, Y_test = train_test_split(phi,y,train_size=750,random_state=52)

    N = len(Y)
    N_test = len(Y_test)


    # set upp things for training
    # if save_start == 0:
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
    # else:
    #     df = pd.read_csv('results/ced_comp/evals.csv')
    #     fcn_train_mse = df['fcn_train_mse']
    #     fcn_test_mse = df['fcn_test_mse']
    #     fcn_num_params = df['fcn_num_params']
    #     ebm_train_mse = df['ebm_train_mse']
    #     ebm_test_mse = df['ebm_test_mse']
    #     ebm_num_params = df['ebm_num_params']
    #     ebm_train_score = df['ebm_train_score']
    #     ebm_test_score = df['ebm_test_score ']
    #     fcn_hidden_dims = df['fcn_hidden_dims']
    #     fcn_layers = df['fcn_layers']
    #     fcn_nonlinearity = df['fcn_nonlinearity']
    #     fcn_batch_size = df['fcn_batch_size']
    #     ebm_feature_dim = df['ebm_feature_dim']
    #     ebm_predictor_dim = df['ebm_predictor_dim']
    #     ebm_batch_size = df['ebm_batch_size']

    for exp in range(save_start,num_exps):
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

        fcn_train_mse.append(mse(yDS.unscale_data(Y), yDS.unscale_data(Y_pred)))
        fcn_test_mse.append(mse(yDS.unscale_data(Y_test), yDS.unscale_data(Y_pred_test)))
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

        ebm_train_mse.append(mse(yDS.unscale_data(Y), yDS.unscale_data(Y_pred_ebm.squeeze().numpy())))
        ebm_test_mse.append(mse(yDS.unscale_data(Y_test), yDS.unscale_data(Y_pred_ebm_test.squeeze().numpy())))
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

        with open('results/ced_comp/fcn'+str(exp)+'.pkl',"wb") as f:
            pickle.dump(net, f)

        with open('results/ced_comp/ebm' + str(exp) + '.pkl', "wb") as f:
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
df.to_csv('results/ced_comp/evals.csv')
df2 = df.dropna()

print("FCN BEST MSE: ", df2['fcn_test_mse'].min())
print("EBM BEST MSE: ", df2['ebm_test_mse'].min())
print('FCN # params: ', df2.loc[df2['fcn_test_mse']==df2['fcn_test_mse'].min(),'fcn_num_params'].to_numpy())
print('EBM # params: ', df2.loc[df2['ebm_test_mse']==df2['ebm_test_mse'].min(),'ebm_num_params'].to_numpy())


