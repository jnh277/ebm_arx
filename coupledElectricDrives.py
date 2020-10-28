import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
import Models
import scipy.linalg as linalg
from sklearn.model_selection import train_test_split

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
    CEDdata = pd.read_csv('./data/coupledElectricDrives/DATAPRBS.csv')

    u = np.reshape(CEDdata[['u1', 'u2', 'u3']].to_numpy().T,(-1,))
    y = np.reshape(CEDdata[['z1', 'z2', 'z3']].to_numpy().T, (-1,))


    # CEDdata = pd.read_csv('./data/coupledElectricDrives/DATAUNIF.csv')
    #
    # u = np.reshape(CEDdata[['u11','u12']].to_numpy().T,(-1,))
    # y = np.reshape(CEDdata[['z11', 'z12']].to_numpy().T, (-1,))

    u_max = u.max()
    u_min = u.min()
    y_max = y.max()
    y_min = y.min()
    y = (y - y_min) / (y_max - y_min)*2-1
    u = (u - u_min) / (u_max - u_min) * 2 - 1

    order = [3, 3]
    max_delay = np.max((order[0],order[1]-1))
    phi = build_phi_matrix(y, order, u)
    y = y[max_delay:]

    ## split randomly
    phi_est,phi_val,yEst,yVal = train_test_split(phi,y,train_size=750,random_state=52)

    # split by experiment
    # phi_est = phi[:(500-max_delay),:]
    # yEst = y[:(500-max_delay)]
    # phi_val = phi[(500-max_delay):,:]
    # yVal = y[(500-max_delay):]


    # uEst = CEDdata['u11'].to_numpy()
    # yEst = CEDdata['z11'].to_numpy()
    #
    # uVal = CEDdata['u12'].to_numpy()
    # yVal = CEDdata['z12'].to_numpy()
    #
    # u_mean = uEst.mean()
    # u_std = uEst.std()
    #
    # y_mean = yEst.mean()
    # y_std = yEst.max()
    #
    # uEst = (uEst - u_mean) / u_std
    # yEst = (yEst - y_mean) / y_std
    #
    # uVal = (uVal - u_mean) / u_std
    # yVal = (yVal - y_mean) / y_std

    # order = [3, 3]
    # max_delay = np.max((order[0],order[1]-1))
    # phi_est = build_phi_matrix(yEst, order, uEst)
    # phi_val = build_phi_matrix(yVal, order, uVal)
    # yEst = yEst[max_delay:]
    # yVal = yVal[max_delay:]

    N = len(yEst)
    N_test = len(yVal)

    batch_size = 128
    learning_rate = 0.001       # increased from 0.001
    num_samples = 512
    num_epochs = 600
    stds = torch.zeros((1, 3))
    stds[0, 0] = 0.1
    stds[0, 1] = 0.2
    stds[0, 2] = 0.8


    dataset = data.TensorDataset(torch.from_numpy(phi_est), torch.from_numpy(yEst))
    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


    network = Models.ARXnet(x_dim=np.sum(order),y_dim=1,hidden_dim=75)
    network.double().to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate,weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

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
            optimizer.step()  # (perform optimization step)x_test

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

    yhat_init = Models.init_predict(X_test, X_test[:,0].clone().detach().unsqueeze(1), network, 2028,[-1.0,1.0])
    yhat = yhat_init.clone()
    # yhat = Y_test.clone().detach().unsqueeze(1)
    # yhat = torch.zeros((N-1,))

    # refining the prediction
    yhat.requires_grad = True
    pred_optimizer = torch.optim.Adam([yhat], lr=0.001)

    max_steps = 200
    #
    score_save = []
    score_save2 = np.zeros((len(yhat),max_steps))
    for step in range(max_steps):
        score = network(X_test,yhat)
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

    # if min(abs(yhat-X_test[:,0])) < 1e-8:
    #     print('prediction possible failed to converge')

    ##### investigate prediction with low likelihood
    # ind = np.where(abs(yhat-X_test[:,0]).detach().numpy()==min(abs(yhat-X_test[:,0])).detach().numpy())
    e = (yhat.squeeze() - Y_test).detach().numpy()
    # ind = np.where(abs(e)==max(abs(e)))
    ind = [125,1]
    res = 300
    x_test = X_test[ind[0],:]*torch.ones(res,1).double()
    y_test = torch.linspace(-1.0,1.0,res).unsqueeze(1).double()

    scores = network(x_test,y_test)
    dt = y_test[1]-y_test[0]
    denom = dt * scores.exp().sum().detach()
    # denom = 1

    plt.plot(y_test.detach(),scores.exp().detach()/denom)
    plt.title("investigating largest error")
    plt.xlabel('X')
    plt.axvline(yhat[ind[0]].detach().numpy()[0],ls='--',color='b')
    plt.axvline(Y_test[ind[0]].detach().numpy(), ls='--', color='r')
    plt.ylabel('p(y|X)')
    plt.legend(['learned density','predicted point','actual meas'])
    plt.show()
    ##############


    plt.plot(Y_test.detach())
    plt.plot(yhat.detach())
    plt.plot(lsq_pred,ls='--')
    plt.legend(['Meausrements','Predictions','baseline'])
    plt.title('Coupled electric drives validation data')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.show()

    e = yhat.squeeze() - Y_test
    e2 = lsq_pred - yVal
    mse = torch.mean((yhat.squeeze() - Y_test)**2)
    print('Test MSE')
    print('Least squares', mse_baseline)
    print('EBM NN:', mse.item())

    plt.hist(e.detach(),bins=30,alpha=0.3)
    plt.hist(e2,bins=30,alpha=0.3)
    plt.show()

