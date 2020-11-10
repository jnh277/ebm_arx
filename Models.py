import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import numpy as np
from tqdm import tqdm

def gauss_density_centered(x, std):
    return torch.exp(-0.5*(x / std)**2) / (math.sqrt(2*math.pi)*std)

def gmm_density_centered(x, std):
    """
    Assumes dim=-1 is the component dimension and dim=-2 is feature dimension. Rest are sample dimension.
    """
    if x.dim() == std.dim() - 1:
        x = x.unsqueeze(-1)
    elif not (x.dim() == std.dim() and x.shape[-1] == 1):
        raise ValueError('Last dimension must be the gmm stds.')
    return gauss_density_centered(x, std).prod(-2).mean(-1)

def sample_gmm_centered(std, num_samples=1):
    num_components = std.shape[-1]
    num_dims = std.numel() // num_components

    std = std.view(1, num_dims, num_components)

    # Sample component ids
    k = torch.randint(num_components, (num_samples,), dtype=torch.int64)
    std_samp = std[0,:,k].t()

    # Sample
    x_centered = std_samp * torch.randn(num_samples, num_dims)
    prob_dens = gmm_density_centered(x_centered, std)

    prob_dens_zero = gmm_density_centered(torch.zeros_like(x_centered), std)

    return x_centered, prob_dens, prob_dens_zero

class PredictorNet(nn.Module):
    def __init__(self, y_dim, hidden_dim=10):
        super().__init__()

        self.fc1_y = nn.Linear(y_dim, hidden_dim)

        self.fc1_xy = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc2_xy = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_xy = nn.Linear(hidden_dim, hidden_dim)
        self.fc4_xy = nn.Linear(hidden_dim, 1)

    def forward(self, x_feature, y):
        # (x_feature has shape: (batch_size, hidden_dim))
        # (y has shape (batch_size, num_samples)) (num_sampes==1 when running on (x_i, y_i))

        if y.dim() == 1:
            y = y.view(-1,1)

        batch_size, num_samples = y.shape

        # Replicate
        x_feature = x_feature.view(batch_size, 1, -1).expand(-1, num_samples, -1) # (shape: (batch_size, num_samples, hidden_dim))

        # resize to batch dimension
        x_feature = x_feature.reshape(batch_size*num_samples, -1) # (shape: (batch_size*num_samples, hidden_dim))
        y = y.reshape(batch_size*num_samples, -1) # (shape: (batch_size*num_samples, 1))

        y_feature = torch.tanh(self.fc1_y(y)) # (shape: (batch_size*num_samples, hidden_dim))

        xy_feature = torch.cat([x_feature, y_feature], 1) # (shape: (batch_size*num_samples, 2*hidden_dim))

        xy_feature = torch.tanh(self.fc1_xy(xy_feature)) # (shape: (batch_size*num_samples, hidden_dim))
        xy_feature = torch.tanh(self.fc2_xy(xy_feature)) + xy_feature # (shape: (batch_size*num_samples, hidden_dim))
        xy_feature = torch.tanh(self.fc3_xy(xy_feature)) + xy_feature # (shape: (batch_size*num_samples, hidden_dim))
        score = self.fc4_xy(xy_feature) # (shape: (batch_size*num_samples, 1))

        score = score.view(batch_size, num_samples) # (shape: (batch_size, num_samples))

        return score

class FeatureNet(nn.Module):
    def __init__(self, x_dim=1, hidden_dim=10):
        super().__init__()

        self.fc1_x = nn.Linear(x_dim, hidden_dim)
        self.fc2_x = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # (x has shape (batch_size, input_dim))

        x_feature = F.relu(self.fc1_x(x)) # (shape: (batch_size, hidden_dim))
        x_feature = F.relu(self.fc2_x(x_feature)) # (shape: (batch_size, hidden_dim))

        return x_feature

class ScalarNet(nn.Module):
    def __init__(self, hidden_dim=10):
        super(ScalarNet, self).__init__()
        input_dim = 1

        self.feature_net = FeatureNet(input_dim, hidden_dim)
        self.predictor_net = PredictorNet(input_dim, hidden_dim)

    def forward(self, x, y):
        # (x has shape (batch_size, 1))
        # (y has shape (batch_size, num_samples)) (num_sampes==1 when running on (x_i, y_i))

        x_feature = self.feature_net(x) # (shape: (batch_size, hidden_dim))
        return self.predictor_net(x_feature, y)

class ARXnet(nn.Module):
    def __init__(self,x_dim=2, y_dim=1, feature_net_dim=10, predictor_net_dim=10):
        super(ARXnet, self).__init__()

        self.feature_net = FeatureNet(x_dim, feature_net_dim)
        self.predictor_net = PredictorNet(y_dim, predictor_net_dim)

    def forward(self, x, y):
        # (x has shape (batch_size, input_dim))
        # (y has shape (batch_size, num_samples)) (num_sampes==1 when running on (x_i, y_i))

        x_feature = self.feature_net(x) # (shape: (batch_size, hidden_dim))
        return self.predictor_net(x_feature, y)


def NCE_loss(xs,ys,network,stds, num_samples,device=torch.device("cpu")):
    x_features = network.feature_net(xs)  # (shape: (batch_size, hidden_dim))
    scores_gt = network.predictor_net(x_features, ys)  # (shape: (batch_size, 1))
    scores_gt = scores_gt.squeeze(1)  # (shape: (batch_size))

    y_samples_zero, q_y_samples, q_ys = sample_gmm_centered(stds, num_samples=num_samples)
    y_samples_zero = y_samples_zero.to(device)  # (shape: (num_samples, 1))
    y_samples_zero = y_samples_zero.squeeze(1)  # (shape: (num_samples))
    q_y_samples = q_y_samples.to(device)  # (shape: (num_samples))
    y_samples = ys + y_samples_zero.unsqueeze(0)  # (shape: (batch_size, num_samples))          # uncenters
    q_y_samples = q_y_samples.unsqueeze(0) * torch.ones(y_samples.size()).to(device)  # (shape: (batch_size, num_samples))
    q_ys = q_ys[0] * torch.ones(xs.size(0)).to(device)  # (shape: (batch_size))

    scores_samples = network.predictor_net(x_features, y_samples)  # (shape: (batch_size, num_samples))

    ########################################################################
    # compute loss:
    ########################################################################
    loss = -torch.mean(scores_gt - torch.log(q_ys) - torch.log(
        torch.exp(scores_gt - torch.log(q_ys)) + torch.sum(torch.exp(scores_samples - torch.log(q_y_samples)),
                                                           dim=1)))
    return loss


def init_predict(xs, ys, network, num_samples,range_vals, device=torch.device("cpu")):
    x_features = network.feature_net(xs)  # (shape: (batch_size, hidden_dim))

    # this is grid init
    # y_samples_zero = torch.linspace(range_vals[0], range_vals[1], num_samples).unsqueeze(1).double()
    # y_samples_zero = y_samples_zero.to(device)  # (shape: (num_samples, 1))
    # y_samples_zero = y_samples_zero.squeeze(1)  # (shape: (num_samples))
    # y_samples = ys + y_samples_zero.unsqueeze(0)  # (shape: (batch_size, num_samples))          # uncenters
    y_samples = torch.zeros(ys.shape).double()+torch.linspace(range_vals[0], range_vals[1], num_samples).unsqueeze(0).double()
    y_samples = y_samples.to(device)

    scores_samples = network.predictor_net(x_features, y_samples)  # (shape: (batch_size, num_samples))
    inds = scores_samples.argmax(1)
    yhat_init = torch.zeros(ys.shape).double()
    for i in range(ys.shape[0]):
        yhat_init[i] = y_samples[i, inds[i]]
    return yhat_init, y_samples, scores_samples

#
class EBM_ARX_net(object):
    def __init__(self,feature_net_dim: int = 100, predictor_net_dim: int = 100, cpu_only: bool = False,
                 random_state: int = 0, lr: float = 0.001, decay_rate: float = 1.0, use_double: bool = True,
                 stds=torch.tensor([0.1, 0.2, 0.8]), num_samples: int = 512, num_epochs: int = 300,
                 batch_size: int = 128):
        self.net = None
        self.training_losses = None
        self.scaling = None
        self.x_dim = None
        self.y_dim = None
        use_cuda = torch.cuda.is_available() and not cpu_only
        self.device = torch.device('cuda:0' if use_cuda else 'cpu')
        self.random_state = random_state
        self.lr = lr
        self.decay_rate = decay_rate
        self.dtype = torch.double if use_double else torch.float
        self.stds = stds
        self.num_samples = num_samples
        self.num_epochs = num_epochs
        self.feature_net_dim = feature_net_dim
        self.predictor_net_dim = predictor_net_dim
        self.num_samples = num_samples
        self.batch_size = batch_size

    @staticmethod
    def _get_nn(x_dim, y_dim, feature_net_dim, predictor_net_dim):
        net = ARXnet(x_dim=x_dim, y_dim=y_dim, feature_net_dim=feature_net_dim, predictor_net_dim=predictor_net_dim)
        return net

    @staticmethod
    def _NCE_loss(xs ,ys , network, stds, num_samples, device, dtype):
        x_features = network.feature_net(xs)  # (shape: (batch_size, hidden_dim))
        scores_gt = network.predictor_net(x_features, ys)  # (shape: (batch_size, 1))
        scores_gt = scores_gt.squeeze(1)  # (shape: (batch_size))

        y_samples_zero, q_y_samples, q_ys = sample_gmm_centered(stds, num_samples=num_samples)
        y_samples_zero = y_samples_zero.to(dtype=dtype,device=device).squeeze(1)   # (shape: (num_samples))
        q_y_samples = q_y_samples.to(dtype=dtype,device=device)  # (shape: (num_samples))
        y_samples = ys + y_samples_zero.unsqueeze(0)  # (shape: (batch_size, num_samples))          # uncenters
        q_y_samples = q_y_samples.unsqueeze(0) * torch.ones(y_samples.size(),dtype=dtype,device=device)  # (shape: (batch_size, num_samples))
        q_ys = q_ys[0].to(dtype=dtype,device=device) * torch.ones(xs.size(0),dtype=dtype,device=device)  # (shape: (batch_size))

        scores_samples = network.predictor_net(x_features, y_samples)  # (shape: (batch_size, num_samples))

        ########################################################################
        # compute loss:
        ########################################################################
        loss = -torch.mean(scores_gt - torch.log(q_ys) - torch.log(
            torch.exp(scores_gt - torch.log(q_ys)) + torch.sum(torch.exp(scores_samples - torch.log(q_y_samples)),
                                                               dim=1)))
        return loss

    @staticmethod
    def _train(net, optimizer, loader, device, stds, num_samples, dtype):
        net = net.train()
        total_loss = 0
        for step, (xs, ys) in enumerate(loader):
            xs = xs.to(device)  # (shape: (batch_size, 1))
            ys = ys.unsqueeze(1).to(device)  # (shape: (batch_size, 1))
            loss = EBM_ARX_net._NCE_loss(xs, ys, net, stds, num_samples, device, dtype)
            total_loss += loss.data.cpu().numpy()
            ########################################################################
            # optimization step:
            ########################################################################
            optimizer.zero_grad()  # (reset gradients)
            loss.backward()  # (compute gradients)
            optimizer.step()  # (perform optimization step)x_test

        return total_loss / (step+1)

    @staticmethod
    def _gridpredict(xs, network, num_samples, dtype, range_vals=(-1.2, 1.2)):
        x_features = network.feature_net(xs)  # (shape: (batch_size, hidden_dim))
        y_shape = (xs.shape[0], 1)
        xt = torch.linspace(range_vals[0], range_vals[1], num_samples, dtype=dtype)
        y_samples = torch.zeros(y_shape, dtype=dtype) + xt.unsqueeze(0)
        scores_samples = network.predictor_net(x_features, y_samples)  # (shape: (batch_size, num_samples))
        inds = scores_samples.argmax(1)
        yhat = torch.zeros(y_shape, dtype=dtype)
        for i in range(y_shape[0]):
            yhat[i] = y_samples[i, inds[i]]
        return yhat, y_samples, scores_samples, xt

    def fit(self, X, y):
        X = np.atleast_2d(X)
        N, x_dim = X.shape
        y_dim = np.atleast_2d(y).shape[0]
        # scaling = y.abs().max()
        # X = X/scaling
        # Y = Y/scaling
        torch.manual_seed(self.random_state)
        net = self._get_nn(x_dim, y_dim, self.feature_net_dim, self.predictor_net_dim)
        net = net.to(dtype=self.dtype,device=self.device)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        if self.decay_rate < 1.0:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.decay_rate)
        X = torch.from_numpy(X).to(dtype=self.dtype)
        y = torch.from_numpy(y).to(dtype=self.dtype)
        dset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dset, batch_size=self.batch_size, shuffle=True)
        training_losses = []
        for ep in tqdm(range(self.num_epochs), desc='Training EBM net: '):
            _loss = self._train(net, optimizer, loader, self.device, self.stds, self.num_samples, self.dtype)
            training_losses.append(_loss)
            if self.decay_rate < 1.0:
                scheduler.step()
        self.net = net.cpu()
        self.training_losses = training_losses
        self.x_dim = x_dim
        self.y_dim = y_dim
        # self.scaling = scaling
        return self


    def predict(self, X):
        X = np.atleast_2d(X)
        X = torch.from_numpy(X).to(dtype=self.dtype)
        # initially predicting over a grid
        yhat_init, _, _, _ = self._gridpredict(X, self.net, 2028, self.dtype)
        yhat = yhat_init.clone()

        # refining the prediction
        yhat.requires_grad = True
        pred_optimizer = torch.optim.Adam([yhat], lr=0.001)
        max_steps = 200

        prediction_score = []
        for step in range(max_steps):
            score = self.net(X, yhat)
            # find the point that maximises the score
            prediction_score.append(score.sum().item())
            neg_score = (-1 * score).sum()
            pred_optimizer.zero_grad()
            neg_score.backward()
            pred_optimizer.step()

        return yhat.detach(), prediction_score

    def pdf_predict(self, X):
        X = torch.from_numpy(X).type(self.dtype)
        _, y_grid, scores_grid, xt = self._gridpredict(X, self.net, 2028, self.dtype)
        dt = xt[1] - xt[0]
        denom = scores_grid.exp().detach().sum(1) * dt
        pdf = scores_grid.exp().detach() / np.reshape(denom,(-1,1))
        cdf = np.cumsum(pdf*dt, axis=1)
        u95 = xt[np.argmin(abs(cdf - 0.975), 1)]
        l95 = xt[np.argmin(abs(cdf - 0.025), 1)]
        u99 = xt[np.argmin(abs(cdf - 0.995), 1)]
        l99 = xt[np.argmin(abs(cdf - 0.005), 1)]
        u65 = xt[np.argmin(abs(cdf - 0.825), 1)]
        l65 = xt[np.argmin(abs(cdf - 0.175), 1)]

        return pdf, cdf, u95, l95, u99, l99, u65, l65