import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm
from chen_arx_example import GenerateChenData


class FullyConnectedNet(object):
    def __init__(self, n_hidden: int = 20,  n_interm_layers: int = 1,
                 nonlinearity: str = 'relu', lr: float = 0.001, momentum: float = 0.95,
                 nesterov: bool = False,
                 epochs: int = 300, batch_size: int = 32, decay_rate: float = 1.0,
                 random_state: int = 0, cpu_only: bool = False, verbose: bool = False):
        self.n_hidden = n_hidden
        self.n_interm_layers = n_interm_layers
        use_cuda = torch.cuda.is_available() and not cpu_only
        self.device = torch.device('cuda:0' if use_cuda else 'cpu')
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.random_state = random_state
        self.nonlinearity = nonlinearity
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.decay_rate = decay_rate
        self.net = None

    @staticmethod
    def get_nn(n_inputs, n_hidden, n_iterm_layers, nonlinearity):
        layers = []
        # Get nonlinerity
        if nonlinearity.lower() == 'relu':
            nl = nn.ReLU(True)
        elif nonlinearity.lower() == 'tanh':
            nl = nn.Tanh()
        else:
            raise ValueError('invalid nonlinearity {}'.format(nonlinearity))
        layers += [nn.Linear(n_inputs, n_hidden), nl]
        for i in range(n_iterm_layers-1):
            layers += [nn.Linear(n_hidden, n_hidden), nl]
        layers += [nn.Linear(n_hidden, 1)]
        net = nn.Sequential(*layers)
        # Initialize modules
        for m in net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity=nonlinearity.lower())
                nn.init.zeros_(m.bias)
        return net

    @staticmethod
    def _train(ep, net, optimizer, loader, n_total, device, verbose=True):
        net = net.train()
        total_loss = 0
        n_entries = 0
        desc = "Epoch {:2d}: train - Loss: {:.6f}"
        if verbose:
            pbar = tqdm(initial=0, leave=True, total=n_total,
                        desc=desc.format(ep, 0), position=0)
        for i, data in enumerate(loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, outputs = data
            inputs = inputs.to(device)
            outputs = outputs.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            predictions = net(inputs)
            loss = nn.functional.mse_loss(predictions.flatten(), outputs.flatten())
            loss.backward()
            optimizer.step()
            # Update
            bs = len(outputs)
            total_loss += loss.detach().cpu().numpy() * bs
            n_entries += bs
            # Update train bar
            if verbose:
                pbar.desc = desc.format(ep, total_loss / n_entries)
                pbar.update(bs)
        if verbose:
            pbar.close()
        return total_loss / n_entries

    @staticmethod
    def _eval(net, loader, n_total, device, verbose=True):
        net.eval()
        n_entries = 0
        predictions_list = []
        if verbose:
            pbar = tqdm(initial=0, leave=True, total=n_total,
                         position=0)
        for i, data in enumerate(loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, = data
            inputs = inputs.to(device)
            with torch.no_grad():
                predictions = net(inputs)
            # Update
            predictions_list.append(predictions)
            bs = len(predictions)
            n_entries += bs
            # Update train bar
            if verbose:
                pbar.update(bs)
        if verbose:
            pbar.close()
        return torch.cat(predictions_list).detach().cpu().flatten().numpy()

    def fit(self, X, y):
        X = np.atleast_2d(X)
        n_total, n_in = X.shape
        torch.manual_seed(self.random_state)
        net = self.get_nn(n_in, self.n_hidden, self.n_interm_layers, self.nonlinearity)
        net.to(self.device)
        optimizer = optim.SGD(net.parameters(), lr=self.lr, momentum=self.momentum, nesterov=self.nesterov)
        if self.decay_rate < 1.0:
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, self.decay_rate)
        X = torch.from_numpy(X).to(dtype=torch.float32)
        y = torch.from_numpy(y).to(dtype=torch.float32)
        dset = torch.utils.data.TensorDataset(X, y)
        loader = DataLoader(dset, batch_size=32, shuffle=True)
        for ep in range(self.epochs):
            _loss = self._train(ep, net, optimizer, loader,
                                n_total, self.device, self.verbose)
            if self.verbose:
                for param_group in optimizer.param_groups:
                    current_lr = param_group['lr']
                tqdm.write('Train loss : {:.6f},  Lr: {:.6f}'.format(_loss, current_lr))
            if self.decay_rate < 1.0:
                scheduler.step()
        self.net = net
        return self

    def predict(self, X):
        X = np.atleast_2d(X)
        n_total, n_features = X.shape
        X = torch.from_numpy(X).to(dtype=torch.float32)
        if n_total < self.batch_size:
            y = self.net(X).detach().cpu().flatten().numpy()
        else:
            dset = torch.utils.data.TensorDataset(X)
            loader = DataLoader(dset, batch_size=self.batch_size, shuffle=False)
            y = self._eval(self.net, loader, n_total, self.device, self.verbose)
        return y

    def __repr__(self):
        return '{}({},{},{},{},{},{},{},{},{},{})'.format(
            type(self).__name__, self.n_features, self.n_interm_layers,
            self.nonlinearity, self.lr, self.momentum, self.nesterov, self.epochs, self.batch_size,
            self.decay_rate, self.random_state)


def evaluate(mdl, X_train, z_train, X_test, z_test):
    # One-step-ahead prediction
    z_pred_train = mdl.predict(X_train)
    z_pred_test = mdl.predict(X_test)
    return mse(z_train, z_pred_train), mse(z_test, z_pred_test), z_pred_train, z_pred_test


def mse(y_true, y_mdl):
    return np.mean((y_true - y_mdl)**2)


# ---- Main script ----
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    N = 2000
    N_test = 500
    hidden_dim = 100
    batch_size = 128
    learning_rate = 0.001
    num_samples = 512
    num_epochs = 100
    stds = torch.zeros((1, 3))
    stds[0, 0] = 0.2
    stds[0, 1] = 0.4
    stds[0, 2] = 1.0
    noise_form = 'gaussian'
    save_results = True
    path = 'fc.pt'

    torch.manual_seed(117)
    np.random.seed(117)

    dataGen = GenerateChenData(noise_form=noise_form, sd_v=0.3, sd_w=0.3)
    X, Y, _, _ = dataGen(N, 1)
    X_test, Y_test, _, _ = dataGen(N_test, 1)

    net = FullyConnectedNet(verbose=True)
    net = net.fit(X, Y)

    Y_pred = net.predict(X)
    Y_pred_test = net.predict(X_test)
    print(mse(Y, Y_pred), mse(Y_test, Y_pred_test))

    fig, ax = plt.subplots()
    k = range(len(Y_test))
    ax.plot(k, Y_test, color='green', label='observed')
    ax.plot(k, Y_pred_test, color='blue', label='predicted')
    ax.set_xlabel('k')
    ax.set_ylabel('y')
    ax.legend()
    plt.show()
