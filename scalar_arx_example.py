# uses an arx model structure to learn a dynamic systems output probability density function
# this example is for a very simple no input single output feedback system
# what is interesting is that it can learn different pdfs: gaussian, cauchy, bimodal gaussian

import torch
import matplotlib.pyplot as plt
import torch.utils.data as data
import Models
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")      # use gpu if available

N = 6000
batch_size = 64
learning_rate = 0.001
num_samples = 1024
num_epochs = 100
stds = torch.zeros((1, 2))
# worked well for bimodal
stds[0, 0] = 0.4
stds[0, 1] = 0.6

noise_form = 'bimodal'            # this can also be 'gaussian', or 'bimodal'

# simulate a really simple arx system
a = 0.95
# sig_m = 0.1     # works well for bimodal
sig_m = 0.2
y0 = 2

y = torch.empty((N,))
y[0] = y0


if noise_form == 'bimodal': # bimodal noise
    for i in range(N-1):
        # y[i+1] = a*y[i]+ sig_m*torch.randn((1,))
        if torch.rand((1,)) > 0.5:
            y[i + 1] = a * y[i] + 0.4 + sig_m * torch.randn((1,))
        else:
            y[i + 1] = a * y[i] - 0.4 + sig_m * torch.randn((1,))
elif noise_form == 'cauchy': # cauchy noise
    for i in range(N-1):
        y[i+1] = a*y[i]+ sig_m*torch.from_numpy(np.random.standard_t(1, (1,)))

elif noise_form == 'gaussian': # gaussian
    for i in range(N-1):
        y[i+1] = a*y[i] + torch.from_numpy(np.random.normal(0,sig_m,(1,)))

# normalise the data
y = y/2

plt.hist(y[1:]-a*y[:-1],bins=30, range=(-.5,0.5))
plt.title('noise distribution')
plt.show()

# convert into ML inputs and outputs
X = y[:-1]
Y = y[1:]



dataset = data.TensorDataset(X,Y)
train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)




network = Models.ScalarNet(hidden_dim=50)

network.to(device)

optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

epoch_losses_train = []

for epoch in range(num_epochs):
    network.train()  # (set in training mode, this affects BatchNorm and dropout)
    batch_losses = []
    for step, (xs, ys) in enumerate(train_loader):
        xs = xs.unsqueeze(1).to(device)  # (shape: (batch_size, 1))
        ys = ys.unsqueeze(1).to(device)  # (shape: (batch_size, 1))

        x_features = network.feature_net(xs)  # (shape: (batch_size, hidden_dim))
        scores_gt = network.predictor_net(x_features, ys)  # (shape: (batch_size, 1))
        scores_gt = scores_gt.squeeze(1)  # (shape: (batch_size))

        y_samples_zero, q_y_samples, q_ys = Models.sample_gmm_centered(stds, num_samples=num_samples)
        y_samples_zero = y_samples_zero.to(device)  # (shape: (num_samples, 1))
        y_samples_zero = y_samples_zero.squeeze(1)  # (shape: (num_samples))
        q_y_samples = q_y_samples.to(device)  # (shape: (num_samples))
        y_samples = ys + y_samples_zero.unsqueeze(0)  # (shape: (batch_size, num_samples))          # uncenters
        q_y_samples = q_y_samples.unsqueeze(0) * torch.ones(y_samples.size()).to(device)  # (shape: (batch_size, num_samples))
        q_ys = q_ys[0] * torch.ones(xs.size(0))  # (shape: (batch_size))
        q_ys = q_ys.to(device)
        scores_samples = network.predictor_net(x_features, y_samples)  # (shape: (batch_size, num_samples))

        ########################################################################
        # compute loss:
        ########################################################################
        loss = -torch.mean(scores_gt - torch.log(q_ys) - torch.log(
            torch.exp(scores_gt - torch.log(q_ys)) + torch.sum(torch.exp(scores_samples - torch.log(q_y_samples)),
                                                               dim=1)))

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

x_test = 0*torch.ones((100,1))
y_test = torch.linspace(-0.5,0.5,100).unsqueeze(1)

scores = network(x_test,y_test)

# the x2 on the x axis is to undo the scaling performed that the beginning of script
plt.plot(2*y_test.detach(),scores.exp().detach()/1.175e7,linewidth=3)
plt.hist(2*(y[1:]-a*y[:-1]),bins=30, range=(-1.0,1.0))
plt.title("noise distribution")
plt.legend(['learned distribution','true distribution'])
plt.show()


# how to make predictions yhat??
yhat = X.clone().detach()
# yhat = torch.zeros((N-1,))
yhat.requires_grad = True
pred_optimizer = torch.optim.Adam([yhat], lr=0.01)
max_steps = 1000
#
for step in range(max_steps):
    score = network(X.unsqueeze(1),yhat.unsqueeze(1))
    # find the point that maximises the score
    neg_score = (-1*score).sum()
    pred_optimizer.zero_grad()
    neg_score.backward()
    pred_optimizer.step()


plt.plot(Y.detach())
plt.plot(yhat.detach())
plt.legend(['Meausrements','Predictions'])
plt.show()