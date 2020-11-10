# data generated using a second order ARX model
import numpy as np
import torch
import matplotlib.pyplot as plt
import Models
import scipy.linalg as linalg
import pickle5 as pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")      # use gpu if available


class GenerateARXData(object):
    def __init__(self, y0=[0,0], sd_u=0.1, sd_y=0.3, noise_form='gaussian'):
        self.sd_u = sd_u
        self.sd_v = sd_y
        self.y0 = y0
        self.noise_form = noise_form

    def _generate_random_input(self, n):
        u = np.random.normal(0, self.sd_u, (n,1))
        return u

    def _noise_model(self):
        if self.noise_form == 'clipped_gaussian':
            e = max(min(self.sd_v*1.5,self.sd_v * np.random.randn()),-self.sd_v*1.5)
        elif self.noise_form == 'bimodal':
            if np.random.uniform() > 0.5:
                e = self.sd_v * np.random.randn() + self.sd_v*2
            else:
                e = self.sd_v * np.random.randn() - self.sd_v*2
        elif self.noise_form == 'upper_clipped_gaussian':
            e = min(self.sd_v*1.5,self.sd_v * np.random.randn())
        elif self.noise_form == 'half_gaussian':
            e = abs(self.sd_v * np.random.randn())
        elif self.noise_form == 'cauchy':
            e = self.sd_v * torch.from_numpy(np.random.standard_t(1, (1,)))
        else:
            e = self.sd_v * np.random.randn()

        return e

    def _linear_function(self, y1, y2, u1, u2):
        return (1.5* y1 - 0.7 * y2 + u1 + 0.5 * u2)

    def _simulate_system(self, u, n):
        y = np.zeros((n,1))
        e = np.zeros((n,1))
        y[0] = self.y0[0]
        y[1] = self.y0[1]

        for k in range(2, n):
            e[k] = self._noise_model()
            y[k] = self._linear_function(y[k - 1], y[k - 2], u[k - 1], u[k - 2]) + e[k]
        return y, e

    def __call__(self, sequence_length, reps):


        X = np.zeros((0,4))
        Y = np.zeros((0,))
        E = np.zeros((0,))

        for i in range(reps):
            u = self._generate_random_input(sequence_length)
            y, e = self._simulate_system(u,sequence_length)

            X = np.concatenate((X, np.hstack((y[:-2],y[1:-1],u[:-2],u[1:-1]))))
            Y = np.concatenate((Y, y[2:,0]))
            E = np.concatenate((E, e[2:,0]))
        return X, Y, E


# ---- Main script ----
if __name__ == "__main__":
    N = 1000
    N_test = 200
    noise_form = 'gaussian'
    save_results = False
    hidden_dim = 100

    np.random.seed(117)      # set numpy seed to get consistent data
    dataGen = GenerateARXData(noise_form=noise_form)
    X, Y, E = dataGen(N, 1)

    # Scale the data
    scale = Y.max()
    X = X / scale
    Y = Y / scale
    E = E/scale

    # simulate test data set
    X_test, Y_test, _ = dataGen(N_test, 1)
    X_test = X_test/scale
    Y_test = Y_test/scale

    net = Models.EBM_ARX_net(feature_net_dim=hidden_dim,predictor_net_dim=hidden_dim, decay_rate=0.99, num_epochs=150, use_double=False)
    net.fit(X, Y)
    training_losses = net.training_losses

    plt.plot(training_losses)
    plt.title('Training loss')
    plt.xlabel('epoch')
    plt.show()

    # make baseline predictions of test data set using least squares
    estim_param, _resid, _rank, _s = linalg.lstsq(X, Y)
    mse_baseline = np.mean((X_test @ estim_param - Y_test) ** 2)


    # # make predictions of test data set using trained EBM NN
    yhat, prediction_scores = net.predict(X_test)

    plt.plot(prediction_scores[:100])
    plt.title('score during prediction stage')
    plt.show()
    #
    #
    plt.plot(Y_test)
    plt.plot(yhat.detach())
    plt.legend(['Meausrements','Predictions'])
    plt.title('Test set predictions')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.show()

    pdf, cdf, u95, l95, u99, l99, u65, l65, xt = net.pdf_predict(X_test)

    plt.plot(scale * Y_test, color='red', ls='None', marker='*')
    plt.fill_between(np.arange(len(Y_test)), scale*u99, scale*l99, alpha=0.1, color='b')
    plt.fill_between(np.arange(len(Y_test)), scale*u95, scale*l95, alpha=0.1, color='b')
    plt.fill_between(np.arange(len(Y_test)), scale*u65, scale*l65, alpha=0.1, color='b')
    plt.xlabel('t', fontsize=20)
    plt.ylabel('y', fontsize=20)
    plt.xlim([50, 60])
    plt.legend(['measured', 'predicted $p(Y_t=y_t | X_t = x_t$'])
    plt.show()

    if save_results:
        data = {"hidden_dim":hidden_dim,
                "scale":scale,
                "X":X,
                "Y":Y,
                "X_test":X_test,
                "Y_test":Y_test}
        with open('results/arx_example/data.pkl',"wb") as f:
            pickle.dump(data,f)

        with open('results/arx_example/network.pkl',"wb") as f:
            pickle.dump(net, f)

# with open('results/arx_example/network.pkl','rb') as f:
#     net2 = pickle.load(f)