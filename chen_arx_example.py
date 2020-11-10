# data generated using a second order ARX model
import pickle5 as pickle
import numpy as np
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
import Models
import scipy.linalg as linalg

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
    hidden_dim = 400
    batch_size = 128
    learning_rate = 0.001
    num_samples = 512
    num_epochs = 100
    stds = torch.zeros((1, 3))
    stds[0, 0] = 0.2
    stds[0, 1] = 0.4
    stds[0, 2] = 1.0
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
    X = torch.from_numpy(X / scale).double()
    Y = torch.from_numpy(Y / scale).double()
    V = V / scale
    W = W / scale


    # simulate test data set
    X_test, Y_test, _, _ = dataGen(N_test, 1)
    X_test = X_test/scale
    Y_test = Y_test/scale

    dataset = data.TensorDataset(X, Y)
    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


    network = Models.ARXnet(x_dim=4,y_dim=1,feature_net_dim=hidden_dim,predictor_net_dim=hidden_dim)
    network.double().to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

    epoch_losses_train = []

    for epoch in range(num_epochs):
        network.train()  # (set in training mode, this affects BatchNorm and dropout)
        batch_losses = []
        for step, (xs, ys) in enumerate(train_loader):
            xs = xs.to(device) # (shape: (batch_size, 1))
            ys = ys.unsqueeze(1).to(device)  # (shape: (batch_size, 1))

            loss = Models.NCE_loss(xs, ys, network, stds, num_samples, device)

            loss_value = loss.data.cpu().numpy()
            batch_losses.append(loss_value)

            ########################################################################
            # optimization step:
            ########################################################################
            optimizer.zero_grad()  # (reset gradients)
            loss.backward()  # (compute gradients)
            optimizer.step()  # (perform optimization step)

            # print("max_score_samp = {} ,  max_score = {}".format(scores_samples.max().item(), scores_gt.max().item()))

        epoch_loss = np.mean(batch_losses)
        epoch_losses_train.append(epoch_loss)
        print('Epoch: {0} train loss: {1}'.format(epoch,epoch_loss))

    network.cpu()

    plt.plot(epoch_losses_train)
    plt.title('Training loss')
    plt.xlabel('epoch')
    plt.show()

    x_test = 0*torch.ones((100,4)).double()
    y_test = torch.linspace(-2/scale,2/scale,100).unsqueeze(1).double()

    scores = network(x_test,y_test)
    dt = y_test[1]-y_test[0]
    denom = scale*dt * scores.exp().sum().detach()

    plt.hist(scale*(W+V),bins=30,density=True)
    plt.plot(scale*y_test.detach(),scores.exp().detach()/denom)
    plt.title("learned Gaussian distribution")
    plt.legend(['true','learned'])
    plt.xlabel('e')
    plt.ylabel('p(e)')
    plt.show()




    # make baseline predictions of test data set using least squares
    estim_param, _resid, _rank, _s = linalg.lstsq(X.numpy(), Y.numpy())
    rmse_baseline = np.sqrt(np.mean((X_test @ estim_param*scale - Y_test*scale) ** 2))

    # make predictions of test data set using trained EBM NN
    X_test = torch.from_numpy(X_test).double()
    Y_test = torch.from_numpy(Y_test).double()

    yhat = X_test[:,0].clone().detach()
    # yhat = Y_test.clone().detach()
    # yhat = torch.zeros((N-1,))
    yhat.requires_grad = True
    pred_optimizer = torch.optim.Adam([yhat], lr=0.01)
    # pred_optimizer = torch.optim.SGD([yhat],lr=0.01)
    max_steps = 100
    #
    score_save = []
    score_save2 = np.zeros((len(yhat),max_steps))
    for step in range(max_steps):
        score = network(X_test,yhat.unsqueeze(1))
        # find the point that maximises the score
        score_save.append(score.sum().item())
        score_save2[:,step] = score.squeeze().detach()
        neg_score = (-1*score).sum()
        pred_optimizer.zero_grad()
        neg_score.backward()
        pred_optimizer.step()

    diff = (yhat - X_test[:,0]).detach().numpy()
    if any(abs(diff)<1e-10):
        print('not all predictions converged')
    ind = np.where(abs(diff) == min(abs(diff)))

    plt.plot(score_save[:100])
    plt.title('score during prediction stage')
    plt.show()

    plt.plot(score_save2.T)
    plt.show()

    plt.plot(scale*Y_test[:].detach())
    plt.plot(scale*yhat[:].detach())
    plt.legend(['Meausrements','Predictions'])
    plt.title('Test set predictions')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.show()


    e = yhat*scale - Y_test*scale
    #
    plt.plot(abs(e.detach()))
    plt.ylabel('error magnitudes')
    plt.show()

    ind = abs(e) < 4*e.std()
    pytorch_total_params = sum(p.numel() for p in network.parameters())
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
                "X":X.numpy(),
                "Y":Y.numpy(),
                "X_test":X_test.numpy(),
                "Y_test":Y_test.numpy()}
        with open('results/chen_model/data.pkl',"wb") as f:
            pickle.dump(data,f)

        torch.save(network.state_dict(), 'results/chen_model/network.pt')