import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from src.models.vae import SimpleVAE, vae_loss


print("Loading data...")
data = np.load("./data/model_data/dataset.npz")

X_train = data["X_train"]
y_train = data["y_train"]
X_val   = data["X_val"]
y_val   = data["y_val"]
X_test  = data["X_test"]
y_test  = data["y_test"]


X_train_t = torch.from_numpy(X_train).float()

print("Training data shape:", X_train_t.shape)
train_loader = DataLoader(
    TensorDataset(X_train_t),
    batch_size=4,   # keep small, your tensors are huge
    shuffle=False   # time order preserved
)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = SimpleVAE(
    input_shape=X_train_t.shape[1:],  # (3, 2, 544, 320)
    latent_dim=64
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    print(f"Epoch {epoch:02d} ---------------------")
    model.train()
    total_loss = 0

    for (x,) in train_loader:
        x = x.to(device)

        optimizer.zero_grad()
        recon, mu, logvar = model(x)
        loss = vae_loss(recon, x, mu, logvar)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch:02d} | loss = {total_loss / len(X_train):.4e}")
