import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleVAE(nn.Module):
    def __init__(self, input_shape, latent_dim=128):
        super().__init__()

        self.input_shape = input_shape
        self.input_dim = int(torch.prod(torch.tensor(input_shape)))

        # Encoder
        self.fc1 = nn.Linear(self.input_dim, 1024)
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)

        # Decoder
        self.fc2 = nn.Linear(latent_dim, 1024)
        self.fc3 = nn.Linear(1024, self.input_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc2(z))
        return self.fc3(h)

    def forward(self, x):
        x = x.view(x.size(0), -1)   # flatten
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        recon = recon.view(x.size(0), *self.input_shape)
        return recon, mu, logvar


def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl
