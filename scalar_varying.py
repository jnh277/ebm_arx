# data generated using a second order ARX model
import numpy as np
import torch
import matplotlib.pyplot as plt
import Models
import scipy.linalg as linalg
import pickle5 as pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")      # use gpu if available


class GenerateARData(object):
    def __init__(self, y0=2):
        self.y0 = y0

    def _noise_model(self, y1):
        if abs(y1) < 0.5:
            e = 0.3 * np.random.randn()
        else:
            e = 0.05 * np.random.randn()
        # e = 0.3*1/(1+abs(y1))**2*np.random.randn()
        return e

    def _linear_function(self, y1):
        return (0.95 * y1)

    def _simulate_system(self, n):
        y = np.zeros((n,1))
        e = np.zeros((n,1))
        y[0] = self.y0

        for k in range(1, n):
            e[k] = self._noise_model(y[k-1])
            y[k] = self._linear_function(y[k - 1]) + e[k]
        return y, e

    def __call__(self, sequence_length, reps):


        X = np.zeros((0,1))
        Y = np.zeros((0,))
        E = np.zeros((0,))

        for i in range(reps):
            y, e = self._simulate_system(sequence_length)

            X = np.concatenate((X, y[0:-1]))
            Y = np.concatenate((Y, y[1:,0]))
            E = np.concatenate((E, e[1:,0]))
        return X, Y, E


# ---- Main script ----
if __name__ == "__main__":
    N = 1000
    N_test = 200
    noise_form = 'gaussian'
    save_results = False
    hidden_dim = 100

    np.random.seed(117)      # set numpy seed to get consistent data
    dataGen = GenerateARData()
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

    sigma_True = np.zeros(X_test.shape)
    sigma_True[abs(X_test)*scale >= 0.5] = 0.05
    sigma_True[abs(X_test)*scale < 0.5] = 0.3
    mu = 0.95*X_test*scale

    plt.plot(scale * Y_test, color='red', ls='None', marker='*',label='Measured')
    plt.plot(scale * yhat.detach(), color='blue',label='MAP')
    plt.fill_between(np.arange(len(Y_test)), scale*u99, scale*l99, alpha=0.1, color='b',label='Predicted $p(Y_t=y_t | X_t = x_t)$')
    plt.fill_between(np.arange(len(Y_test)), scale*u95, scale*l95, alpha=0.1, color='b')
    plt.fill_between(np.arange(len(Y_test)), scale*u65, scale*l65, alpha=0.1, color='b')
    plt.plot(mu, color='orange', ls='-.', label='True mean')
    plt.plot(mu+2*sigma_True, color='orange', ls='--', label='True +/- 2$\sigma$')
    plt.plot(mu-2*sigma_True, color='orange', ls='--')
    plt.xlabel('t', fontsize=20)
    plt.ylabel('y', fontsize=20)
    plt.xlim([150, 185])
    # plt.legend(['Measured', 'MAP', 'Predicted $p(Y_t=y_t | X_t = x_t)$'])
    plt.legend()
    plt.show()

    # if save_results:
        # data = {"hidden_dim":hidden_dim,
        #         "scale":scale,
        #         "X":X,
        #         "Y":Y,
        #         "X_test":X_test,
        #         "Y_test":Y_test}
        # with open('results/arx_example/data.pkl',"wb") as f:
        #     pickle.dump(data,f)
        #
        # with open('results/arx_example/network.pkl',"wb") as f:
        #     pickle.dump(net, f)

# with open('results/arx_example/network.pkl','rb') as f:
#     net2 = pickle.load(f)