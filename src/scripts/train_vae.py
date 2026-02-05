import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from src.models.vae import LitVAE


print("Loading data...")
data = np.load("/data/rubens/quasi_geostrophic_data/dataset.npz")


X_train = data["X_train"]
X_val   = data["X_val"]
X_test  = data["X_test"]

X_train_t = torch.from_numpy(X_train).float()
X_val_t   = torch.from_numpy(X_val).float()
X_test_t  = torch.from_numpy(X_test).float()


print("Training data shape:", X_train_t.shape)

train_loader = DataLoader(
    TensorDataset(X_train_t),
    batch_size=32,
    shuffle=True,          # shuffle training
    pin_memory=True,
    drop_last=True
)

val_loader = DataLoader(
    TensorDataset(X_val_t),
    batch_size=30,
    shuffle=False,
    pin_memory=True,
    drop_last=False
)

test_loader = DataLoader(
    TensorDataset(X_test_t),
    batch_size=60,
    shuffle=False,
    pin_memory=True,
    drop_last=False
)


device = "cuda" if torch.cuda.is_available() else "cpu"

model = LitVAE(latent_dim=128).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

scaler = GradScaler()   # AMP scaler

tb_logger = TensorBoardLogger(
    save_dir="/data/rubens/quasi_geostrophic_data/models/ConvVAE/lightning_logs",
    name="vae"
)

checkpoint_cb = pl.callbacks.ModelCheckpoint(
    dirpath="/data/rubens/quasi_geostrophic_data/models/ConvVAE/checkpoints",
    save_weights_only=True,
    filename="epoch={epoch:04d}-val={val_loss:.4e}",
    monitor="val_loss",
    mode="min",
    save_top_k=1,          # only keep best model
    every_n_epochs=40,     # check/save every 20 epochs
    save_last=False,
)


trainer = pl.Trainer(
    max_epochs=10000,
    gradient_clip_val=1.0,
    callbacks=[checkpoint_cb],
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    precision="16-mixed",   # AMP
    log_every_n_steps=10,
    logger=tb_logger
)


trainer.fit(model, train_loader, val_loader)
trainer.test(model, test_loader)

