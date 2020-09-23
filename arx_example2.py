import torch
import matplotlib.pyplot as plt
import torch.utils.data as data
import Models
import numpy as np

N = 1000
batch_size = 64
learning_rate = 0.001
num_samples = 1024
num_epochs = 75
stds = torch.zeros((1, 2))
stds[0, 0] = 0.2
stds[0, 1] = 0.4


torch.manual_seed(3689)

# simulate an arx system with 1 input and 1 output feedback
a = 0.95
# b = 0.2
b = 0
sig_m = 0.1

y0 = 2

sig_m = 0.1
sig_m2 = 0.1

y = torch.empty((N,))
y[0] = y0

u = torch.empty((N,))
u[0] = 1


for i in range(N-1):
    if torch.rand((1,)) > 0.8:
        u[i+1] = -1*u[i]
    else:
        u[i+1] = u[i]

    y[i + 1] = a * y[i] + sig_m * torch.randn((1,)) + b * u[i]




# normalise the data
y = y/y.max()
u = u/u.max()


plt.subplot(2,1,1)
plt.plot(y)
plt.title('output')

plt.subplot(2,1,2)
plt.plot(u)
plt.title('input')

plt.show()

# convert into ML inputs and outputs
X = torch.cat([y[:-1].unsqueeze(1),u[:-1].unsqueeze(1)],1)
Y = y[1:]



dataset = data.TensorDataset(X,Y)
train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


network = Models.ARXnet(x_dim=2,y_dim=1,hidden_dim=30)

optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

epoch_losses_train = []

for epoch in range(num_epochs):
    network.train()  # (set in training mode, this affects BatchNorm and dropout)
    batch_losses = []
    for step, (xs, ys) in enumerate(train_loader):
        xs = xs # (shape: (batch_size, 1))
        ys = ys.unsqueeze(1)  # (shape: (batch_size, 1))

        x_features = network.feature_net(xs)  # (shape: (batch_size, hidden_dim))
        scores_gt = network.predictor_net(x_features, ys)  # (shape: (batch_size, 1))
        scores_gt = scores_gt.squeeze(1)  # (shape: (batch_size))

        y_samples_zero, q_y_samples, q_ys = Models.sample_gmm_centered(stds, num_samples=num_samples)
        # y_samples_zero = y_samples_zero.cuda()  # (shape: (num_samples, 1))
        y_samples_zero = y_samples_zero.squeeze(1)  # (shape: (num_samples))
        # q_y_samples = q_y_samples.cuda()  # (shape: (num_samples))
        y_samples = ys + y_samples_zero.unsqueeze(0)  # (shape: (batch_size, num_samples))          # uncenters
        q_y_samples = q_y_samples.unsqueeze(0) * torch.ones(y_samples.size())  # (shape: (batch_size, num_samples))
        q_ys = q_ys[0] * torch.ones(xs.size(0))  # (shape: (batch_size))

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

plt.plot(epoch_losses_train)
plt.show()

x_test = torch.cat([0.3*torch.ones((100,1)),0*torch.ones((100,1))],1)
y_test = torch.linspace(-1,1,100).unsqueeze(1)

scores = network(x_test,y_test)

plt.plot(y_test.detach(),scores.exp().detach())
plt.title("learned Gaussian distribution")
plt.show()


# how to make predictions yhat??
yhat = torch.zeros((N-1,))
yhat.requires_grad = True
pred_optimizer = torch.optim.Adam([yhat], lr=0.01)
max_steps = 1000
#
for step in range(max_steps):
    score = network(X,yhat.unsqueeze(1))
    # find the point that maximises the score
    neg_score = (-1*score).sum()
    pred_optimizer.zero_grad()
    neg_score.backward()
    pred_optimizer.step()

plt.plot(Y.detach())
plt.plot(yhat.detach())
plt.legend(['Meausrements','Predictions'])
plt.show()