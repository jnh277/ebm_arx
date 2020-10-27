import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
import Models
import scipy.linalg as linalg

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



# ---- Main script ----
if __name__ == "__main__":
    tankdata = pd.read_csv('./data/cascadedTanksBenchmark/dataBenchmark.csv')

    uEst = tankdata['uEst'].to_numpy()
    yEst = tankdata['yEst'].to_numpy()
    uVal = tankdata['uVal'].to_numpy()
    yVal = tankdata['yVal'].to_numpy()

    u_mean = uEst.mean()
    u_std = uEst.std()

    y_mean = yEst.mean()
    y_std = yEst.std()

    uEst = (uEst - u_mean) / u_std
    yEst = (yEst - y_mean) / y_std

    uVal = (uVal - u_mean) / u_std
    yVal = (yVal - y_mean) / y_std

    order = [4, 3]
    max_delay = np.max((order[0],order[1]-1))
    phi_est = build_phi_matrix(yEst, order, uEst)
    phi_val = build_phi_matrix(yVal, order, uVal)
    yEst = yEst[max_delay:]
    yVal = yVal[max_delay:]

    N = len(yEst)
    N_test = len(yVal)

    batch_size = 128
    learning_rate = 0.001
    num_samples = 512
    num_epochs = 400
    stds = torch.zeros((1, 2))
    stds[0, 0] = 0.1
    stds[0, 1] = 0.2


    dataset = data.TensorDataset(torch.from_numpy(phi_est), torch.from_numpy(yEst))
    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


    network = Models.ARXnet(x_dim=np.sum(order),y_dim=1,hidden_dim=75)
    network.double().to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.1)

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

        scheduler.step()
        epoch_loss = np.mean(batch_losses)
        epoch_losses_train.append(epoch_loss)
        print('Epoch: {0} train loss: {1}'.format(epoch,epoch_loss))

    network.cpu()

    plt.plot(epoch_losses_train)
    plt.title('Training loss')
    plt.xlabel('epoch')
    plt.show()

    x_test = 0*torch.ones((100,np.sum(order))).double()
    y_test = torch.linspace(-0.1,0.1,100).unsqueeze(1).double()

    scores = network(x_test,y_test)
    dt = y_test[1]-y_test[0]
    denom = dt * scores.exp().sum().detach()

    plt.plot(y_test.detach(),scores.exp().detach()/denom)
    plt.title("learned distribution")
    plt.xlabel('X')
    plt.ylabel('p(y|X)')
    plt.show()


    # make baseline predictions of test data set using least squares
    estim_param, _resid, _rank, _s = linalg.lstsq(phi_est, yEst)
    lsq_pred = phi_val @ estim_param
    mse_baseline = np.mean((lsq_pred - yVal) ** 2)

    # make predictions of test data set using trained EBM NN
    X_test = torch.from_numpy(phi_val).double()
    Y_test = torch.from_numpy(yVal).double()

    yhat = X_test[:,0].clone().detach()
    # yhat = torch.zeros((N-1,))
    yhat.requires_grad = True
    pred_optimizer = torch.optim.Adam([yhat], lr=0.01)

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

    plt.plot(score_save)
    plt.title('Prediction log likelihood')
    plt.show()

    # min(abs(yhat-X_test[:,0]))

    plt.plot(Y_test.detach())
    plt.plot(yhat.detach())
    plt.plot(lsq_pred,ls='--')
    plt.legend(['Meausrements','Predictions','baseline'])
    plt.title('cascaded tanks validation data')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.show()

    mse = torch.mean((yhat - Y_test)**2)
    print('Test MSE')
    print('Least squares', mse_baseline)
    print('EBM NN:', mse.item())

