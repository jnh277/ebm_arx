# η = {a1:na , b0:nb , σe}. = {−1.5,0.7,0,1,0.5,1}
# {0.33,0.98,0.98,0.33,1.22,0.56,−0.18,0.5}
# {0.02, 0.04, 0.02, −1.56, 0.64, 0.05}

import numpy as np
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
import Models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")      # use gpu if available


class GenerateARXData(object):
    def __init__(self, y0=[0,0], sd_u=0.1, sd_y=0.3):
        self.sd_u = sd_u
        self.sd_v = sd_y
        self.y0 = y0

    def _generate_random_input(self, n):
        u = np.random.normal(0, self.sd_u, (n,1))
        return u

    def _linear_function(self, y1, y2, u1, u2):
        return (1.5* y1 - 0.7 * y2 + u1 + 0.5 * u2)

    def _simulate_system(self, u, n):
        y = np.zeros((n,1))
        e = np.zeros((n,1))
        y[0] = self.y0[0]
        y[1] = self.y0[1]

        for k in range(2, n):
            e[k] = self.sd_v * np.random.randn()
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
    N = 6000
    batch_size = 128
    learning_rate = 0.001
    num_samples = 512
    num_epochs = 100
    stds = torch.zeros((1, 2))
    stds[0, 0] = 0.2
    stds[0, 1] = 0.4

    dataGen = GenerateARXData()
    X, Y, E = dataGen(N, 1)

    # Normalise the data
    X = torch.from_numpy(X / X.std(0)).float()
    Y = torch.from_numpy(Y / Y.std(0)).float()

    dataset = data.TensorDataset(X, Y)
    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


    network = Models.ARXnet(x_dim=4,y_dim=1,hidden_dim=50)
    network.to(device)
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
    plt.show()

    x_test = 0*torch.ones((100,4))
    y_test = torch.linspace(-1,1,100).unsqueeze(1)

    scores = network(x_test,y_test)
    dt = y_test[1]-y_test[0]
    denom = dt * scores.exp().sum().detach()

    plt.hist(E,bins=30,density=True)
    plt.plot(y_test.detach(),scores.exp().detach()/denom)
    plt.title("learned Gaussian distribution")
    plt.show()