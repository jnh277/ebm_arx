# uses an arx model structure to learn a dynamic systems output probability density function
# this example is for a very simple no input single output feedback system
# what is interesting is that it can learn different pdfs: gaussian, cauchy, bimodal gaussian

import torch
import matplotlib.pyplot as plt
import torch.utils.data as data
import Models
import numpy as np
import scipy.stats as stats





device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")      # use gpu if available

N = 10000
batch_size = 64
learning_rate = 0.001
num_samples = 1024
num_epochs = 50
stds = torch.zeros((1, 3))
# worked well for bimodal
stds[0, 0] = 0.4
stds[0, 1] = 0.6
stds[0, 2] = 1.0

noise_form = 'gaussian'            # this can be 'gaussian', or 'bimodal', or 'cauchy'

# simulate a really simple arx system
a = 0.95
y0 = 2

y = torch.empty((N,))
y[0] = y0
e = torch.zeros((N,))

if noise_form == 'bimodal': # bimodal noise
    sig_m = 0.1
    torch.manual_seed(19)  # reasonablygood 19,20
    for i in range(N-1):
        if torch.rand((1,)) > 0.5:
            e[i+1] = + 0.4 + sig_m * torch.randn((1,))
        else:
            e[i + 1] = - 0.4 + sig_m * torch.randn((1,))
        y[i + 1] = a * y[i] + e[i+1]
elif noise_form == 'cauchy': # cauchy noise
    sig_m = 0.2
    torch.manual_seed(19)  # 19 is ok
    # worked well for bimodal
    for i in range(N-1):
        e[i+1] = sig_m*torch.from_numpy(np.random.standard_t(1, (1,)))
        y[i+1] = a*y[i]+ e[i+1]

elif noise_form == 'gaussian': # gaussian
    sig_m = 0.2
    torch.manual_seed(13)  # for reproducibility
    for i in range(N-1):
        e[i+1] = torch.from_numpy(np.random.normal(0,sig_m,(1,)))
        y[i+1] = a*y[i] + e[i+1]
elif noise_form == 'bounded_gauss':
    sig_m = 0.2
    for i in range(N-1):
        e[i + 1] = max(-0.3,min(0.3,torch.from_numpy(np.random.normal(0, sig_m, (1,)))))
        y[i + 1] = a * y[i] + e[i + 1]

# normalise the data
scale = 2.0
y = y/scale
e = e/scale
# plt.hist(y[1:]-a*y[:-1],bins=30, range=(-.5,0.5))
# plt.title('noise distribution')
# plt.show()

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
        # compute loss: NCE loss (eq 12) in how train your EBM model
        ########################################################################
        loss = -torch.mean(scores_gt - torch.log(q_ys) - torch.log(
            torch.exp(scores_gt - torch.log(q_ys)) + torch.sum(torch.exp(scores_samples - torch.log(q_y_samples)),
                                                               dim=1)))
        # if torch.isnan(loss):
        #     print('bad')

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

x0 = 0.0
x_test = x0*torch.ones((500,1))
y_test = torch.linspace(-0.5,0.5,500).unsqueeze(1)

scores = network(x_test,y_test)
dt = y_test[1]-y_test[0]
denom = scale*dt * scores.exp().sum().detach()
# denom = 1.0

# the x2 on the x axis is to undo the scaling performed that the beginning of script
# plt.hist(scale*(e+x0),bins=30, range=(-1.0,1.0), density=True)

if noise_form == 'gaussian':
    p_true = stats.norm(0, sig_m).pdf(scale * y_test.detach())
    plt.ylim([0,2.05])
elif noise_form == 'bimodal':
    p_true = 0.5*stats.norm(0.4, sig_m).pdf(scale * y_test.detach())+0.5*stats.norm(-0.4, sig_m).pdf(scale * y_test.detach())
    plt.ylim([0,2.05])
elif noise_form == 'cauchy':
    x_test = x0 * torch.ones((500, 1))
    y_test = torch.linspace(-1.5, 1.5, 500).unsqueeze(1)
    scores = network(x_test, y_test)
    dt = y_test[1] - y_test[0]
    denom = scale * dt * scores.exp().sum().detach()
    p_true = stats.cauchy(0.0, sig_m).pdf(scale * y_test.detach())
    plt.ylim([0,2.05])
plt.plot(scale*y_test.detach(),p_true,linewidth=4)
plt.fill_between(scale*y_test.squeeze().detach(),p_true.squeeze(),p_true.squeeze()*0,color='blue',alpha=0.3)
plt.plot(scale*y_test.detach(),scores.exp().detach()/denom,linewidth=4,ls='--')
plt.xlabel('$e_t$',fontsize=20)
plt.ylabel('$p(e_t)$',fontsize=20)
# plt.title("noise distribution")
plt.legend(['true distribution','Learned distribution'])
plt.xlim([-1,1])
plt.show()


# how to make predictions yhat??
yhat = X.clone().detach()
# yhat = torch.zeros((N-1,))
yhat.requires_grad = True
pred_optimizer = torch.optim.Adam([yhat], lr=0.01)
max_steps = 100
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

yhat_init, yhat_samples, scores_samples = Models.init_predict(Y[:49].unsqueeze(1).double(), Y[:49].clone().detach().double().unsqueeze(1), network.double(), 2028, [-1.0, 1.0])

plt.contour(np.arange(1,50),scale*np.linspace(-1,1,2028),scores_samples.exp().detach().numpy().T,30)
plt.plot(scale*Y[1:50].detach(),color='red',ls='None',marker='*')
plt.xlabel('t',fontsize=20)
plt.ylabel('y',fontsize=20)
plt.xlim([15,50])
plt.legend(['measured','predicted $p(Y_t=y_t | X_t = x_t$'])
plt.show()