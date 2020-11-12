# data generated using the chen model
import pickle5 as pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import Models
import scipy.linalg as linalg
from Models import FullyConnectedNet
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")      # use gpu if available


class GenerateChenData(object):
    def __init__(self, y0=[0,0], sd_u=1.0, sd_v=0.1, sd_w=0.3, noise_form='gaussian'):
        self.sd_u = sd_u
        self.sd_v = sd_v
        self.sd_w = sd_w
        self.y0 = y0
        self.noise_form = noise_form

    def _generate_random_input(self, n):
        u = np.random.normal(0, self.sd_u, (n,1))
        return u

    def _noise_model(self, sigma):
        if self.noise_form == 'clipped_gaussian':
            e = max(min(sigma*1.5,sigma * np.random.randn()),-sigma*1.5)
        elif self.noise_form == 'bimodal':
            if np.random.uniform() > 0.5:
                e = sigma * np.random.randn() + sigma*2
            else:
                e = sigma * np.random.randn() - sigma*2
        elif self.noise_form == 'upper_clipped_gaussian':
            e = min(sigma*1.5,sigma * np.random.randn())
        elif self.noise_form == 'half_gaussian':
            e = abs(sigma * np.random.randn())
        elif self.noise_form == 'cauchy':
            e = sigma * torch.from_numpy(np.random.standard_t(1, (1,)))
        else:
            e = sigma * np.random.randn()

        return e

    def _nonlinear_function(self, y1, y2, u1, u2):
        return (0.8 - 0.5 * np.exp(-y1 ** 2)) * y1 - (0.3 + 0.9 * np.exp(-y1 ** 2)) * y2 \
               + u1 + 0.2 * u2 + 0.1 * u1 * u2

    def _simulate_system(self, u, n):
        y = np.zeros((n,1))
        v = np.zeros((n,1))
        w = np.zeros((n,1))
        y[0] = self.y0[0]
        y[1] = self.y0[1]

        for k in range(2, n):
            v[k] = self._noise_model(self.sd_v)
            w[k] = self._noise_model(self.sd_v)
            y[k] = self._nonlinear_function(y[k - 1], y[k - 2], u[k - 1], u[k - 2]) + v[k]
        return y+w, v, w

    def __call__(self, sequence_length, reps):


        X = np.zeros((0,4))
        Y = np.zeros((0,))
        V = np.zeros((0,))
        W = np.zeros((0,))

        for i in range(reps):
            u = self._generate_random_input(int(np.round(sequence_length/5)*5))
            u = u.reshape((-1,5))
            u[:, 1] = u[:, 0]
            u[:, 2] = u[:, 0]
            u[:, 3] = u[:, 0]
            u[:, 4] = u[:, 0]
            u = u.reshape((-1,1))
            y, v, w = self._simulate_system(u,sequence_length)

            X = np.concatenate((X, np.hstack((y[:-2],y[1:-1],u[:-2],u[1:-1]))))
            Y = np.concatenate((Y, y[2:, 0]))
            V = np.concatenate((V, v[2:, 0]))
            W = np.concatenate((W, w[2:, 0]))
        return X, Y, V, W


# ---- Main script ----
if __name__ == "__main__":
    N = 1000
    N_test = 500
    hidden_dim = 100

    noise_form = 'gaussian'
    save_results = False

    torch.manual_seed(117)
    np.random.seed(117)

    sd_v = 0.3
    sd_w = 0.3
    dataGen = GenerateChenData(noise_form=noise_form,sd_v=sd_v,sd_w=sd_w)
    X, Y, V, W = dataGen(N, 1)

    # Normalise the data
    scale = Y.max(0)
    X = X / scale
    Y = Y / scale
    V = V / scale
    W = W / scale


    # simulate test data set
    X_test, Y_test, _, _ = dataGen(N_test, 1)
    X_test = X_test/scale
    Y_test = Y_test/scale

    net_fcn = FullyConnectedNet(n_hidden=150,n_interm_layers=4)
    net_fcn.fit(X, Y)
    yhat_fcn = net_fcn.predict(X_test)

    net = Models.EBM_ARX_net(use_double=False,feature_net_dim=100,predictor_net_dim=50, decay_rate=0.99, num_epochs=600)
    net.fit(X, Y)
    training_losses = net.training_losses

    plt.plot(training_losses)
    plt.title('Training loss')
    plt.xlabel('epoch')
    plt.show()

    # make baseline predictions of test data set using least squares
    estim_param, _resid, _rank, _s = linalg.lstsq(X, Y)
    rmse_baseline = np.sqrt(np.mean((X_test @ estim_param*scale - Y_test*scale) ** 2))

    # # make predictions of test data set using trained EBM NN
    yhat, prediction_scores = net.predict(X_test)

    plt.plot(prediction_scores[:100])
    plt.title('score during prediction stage')
    plt.show()
    #
    #
    plt.plot(Y_test)
    plt.plot(yhat.detach())
    plt.plot(yhat_fcn)
    plt.legend(['Meausrements','Predictions ebm', 'pred fcn'])
    plt.title('Test set predictions')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.show()


    e = yhat.squeeze()*scale - Y_test*scale
    #
    plt.plot(abs(e.detach()))
    plt.ylabel('error magnitudes')
    plt.show()

    ind = abs(e) < 4*e.std()
    pytorch_total_params = sum(p.numel() for p in net.net.parameters())
    print('Total trainable parameters:',pytorch_total_params)
    print('num outliers:',(len(e)-sum(ind)).item())
    rmse = torch.mean((e[ind])**2).sqrt()
    print('Test RMSE')
    print('Least squares', rmse_baseline)
    print('EBM NN:', rmse.item())

    if save_results:
        data = {"hidden_dim":hidden_dim,
                "scale":scale,
                "sd_v":sd_v,
                "sd_w":sd_w,
                "X":X,
                "Y":Y,
                "X_test":X_test,
                "Y_test":Y_test}
        with open('results/chen_model/data.pkl',"wb") as f:
            pickle.dump(data,f)

        with open('results/chen_model/network.pkl',"wb") as f:
            pickle.dump(net, f)

        with open('results/chen_model/fcn.pkl',"wb") as f:
            pickle.dump(net_fcn, f)