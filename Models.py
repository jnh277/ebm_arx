import torch.nn as nn
import torch
import torch.nn.functional as F
import math


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

class ScalarPredictorNet(nn.Module):
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

class ScalarFeatureNet(nn.Module):
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

        self.feature_net = ScalarFeatureNet(input_dim, hidden_dim)
        self.predictor_net = ScalarPredictorNet(input_dim, hidden_dim)

    def forward(self, x, y):
        # (x has shape (batch_size, 1))
        # (y has shape (batch_size, num_samples)) (num_sampes==1 when running on (x_i, y_i))

        x_feature = self.feature_net(x) # (shape: (batch_size, hidden_dim))
        return self.predictor_net(x_feature, y)

class ARXnet(nn.Module):
    def __init__(self,x_dim=2, y_dim=1, hidden_dim=10):
        super(ARXnet, self).__init__()

        self.feature_net = ScalarFeatureNet(x_dim, hidden_dim)
        self.predictor_net = ScalarPredictorNet(y_dim, hidden_dim)

    def forward(self, x, y):
        # (x has shape (batch_size, input_dim))
        # (y has shape (batch_size, num_samples)) (num_sampes==1 when running on (x_i, y_i))

        x_feature = self.feature_net(x) # (shape: (batch_size, hidden_dim))
        return self.predictor_net(x_feature, y)