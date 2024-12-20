import torch
import torch.nn as nn
torch_dtype = torch.float32
torch.set_default_dtype(torch_dtype)
class VAE(nn.Module):
    '''
    Variational Autoencoder (VAE) class.
    This class represents a VAE, which is a type of autoencoder that uses variational inference to train.

    Attributes:
    input_dim (int): The dimension of the input data.
    mid_dim (int): The dimension of the hidden layer.
    features (int): The number of features in the latent space.
    output_layer (nn.Module): The output layer function.

    Methods:
    reparametrize(mu, log_var): Reparameterization trick to sample from the latent space.
    forward(x): Forward pass through the network.
    '''
    def __init__(self, input_dim, mid_dim, features, output_layer=nn.ReLU):
        super().__init__()
        self.input_dim = input_dim
        self.mid_dim = mid_dim
        self.features = features
        self.output_layer = output_layer
        # print('input_dim =\t',input_dim)
        # print('mid_dim =\t',mid_dim)
        # print('features =\t',features)
        self.encoder = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=self.mid_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.mid_dim, out_features=self.features * 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=self.features, out_features=self.mid_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.mid_dim, out_features=self.input_dim),
            # Output activation layer function:
            # activation_layer()
            output_layer()
        )

    def reparametrize(self, mu, log_var):

        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            sample = mu + (eps * std)
        else:
            sample = mu
        return sample

    def forward(self, x):

        mu_logvar = self.encoder(x).view(-1, 2, self.features)
        mu = mu_logvar[:, 0, :]
        log_var = mu_logvar[:, 1, :]

        z = self.reparametrize(mu, log_var)
        reconstruction = self.decoder(z)
        # print('Inside VAE forward')
        # print('reconstruction.shape =\t', reconstruction.shape)
        # print('mu.shape =\t', mu.shape)
        # print('log_var.shape =\t', log_var.shape)
        # print('z.shape =\t', z.shape)
        return reconstruction, mu, log_var, z