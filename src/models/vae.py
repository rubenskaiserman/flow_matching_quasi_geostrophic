import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from torch.cuda.amp import autocast


class ConvVAE(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()

        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(6, 32, 4, 2, 1),   # 544×320 → 272×160
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # 136×80
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), # 68×40
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),# 34×20
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(256 * 34 * 20, latent_dim)
        self.fc_logvar = nn.Linear(256 * 34 * 20, latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, 256 * 34 * 20)

        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # 68×40
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 136×80
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   # 272×160
            nn.ReLU(),
            nn.ConvTranspose2d(32, 6, 4, 2, 1),    # 544×320
        )

    def forward(self, x):
        x = x.view(x.size(0), 6, 544, 320)
        h = self.enc(x)
        h = h.view(x.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        h = self.fc_dec(z)
        h = h.view(x.size(0), 256, 34, 20)
        recon = self.dec(h)
        return recon, mu, logvar


class LitVAE(pl.LightningModule):
    def __init__(self, latent_dim=64, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.model = ConvVAE(latent_dim=latent_dim)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        (x,) = batch
        x = x.view(x.size(0), 6, 544, 320)

        recon, mu, logvar = self(x)
        
        beta = min(1e-2, self.current_epoch / 50 * 1e-2)
        loss, recon_loss, kl = vae_loss(recon, x, mu, logvar, beta)
        
        
        self.log("train/recon", recon_loss, on_epoch=True, prog_bar=True)
        self.log("train/kl", kl, on_epoch=True)
        self.log("train/beta", beta, on_epoch=True)


        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=x.size(0),
        )

        return loss

    def validation_step(self, batch, batch_idx):
        (x,) = batch
        x = x.view(x.size(0), 6, 544, 320)

        recon, mu, logvar = self(x)
        beta = min(1e-2, self.current_epoch / 50 * 1e-2)
        loss, recon_loss, kl = vae_loss(recon, x, mu, logvar, beta)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val/recon", recon_loss, prog_bar=True)
        self.log("val/kl", kl, prog_bar=True)

    def test_step(self, batch, batch_idx):
        (x,) = batch
        x = x.view(x.size(0), 6, 544, 320)

        recon, mu, logvar = self(x)
        beta = min(1e-2, self.current_epoch / 50 * 1e-2)
        loss, recon_loss, kl = vae_loss(recon, x, mu, logvar, beta)

        self.log("test_loss", loss)
        self.log("test/recon", recon_loss)
        self.log("test/kl", kl)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)



def vae_loss(recon_x, x, mu, logvar, beta):
    recon_loss = F.mse_loss(recon_x, x, reduction="mean")
    kl = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp(),
        dim=1
    ).mean()
    
    loss = recon_loss + beta * kl
 
    return loss, recon_loss, kl