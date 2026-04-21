import os
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.models as models


BATCH_SIZE = 128
EPOCHS = 5
LR = 0.001
LAMBDA = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("checkpoints", exist_ok=True)


# -------------------------
# Entropy
# -------------------------
def entropy(p):
    p = torch.clamp(p, 1e-12, 1.0)
    return -torch.sum(p * torch.log2(p), dim=1)


# -------------------------
# Dataset
# -------------------------
class Data(Dataset):
    def __init__(self):
        transform = transforms.Compose([transforms.ToTensor()])

        self.cifar = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )

        self.soft = np.load("data/cifar10h-probs.npy")

    def __len__(self):
        return len(self.cifar)

    def __getitem__(self, i):
        x, _ = self.cifar[i]
        y = torch.tensor(self.soft[i], dtype=torch.float32)
        return x, y


# -------------------------
# Model
# -------------------------
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.resnet18(weights=None)
        self.net.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.net.maxpool = nn.Identity()
        self.net.fc = nn.Linear(self.net.fc.in_features, 10)

    def forward(self, x):
        return self.net(x)


# -------------------------
# Loss
# -------------------------
def custom_loss(logits, target):
    log_probs = torch.log_softmax(logits, dim=1)
    probs = torch.softmax(logits, dim=1)

    # KL
    kl = torch.sum(target * (torch.log(target + 1e-12) - log_probs), dim=1)

    # entropy difference
    h_true = entropy(target)
    h_pred = entropy(probs)

    entropy_loss = torch.abs(h_true - h_pred)

    return (kl + LAMBDA * entropy_loss).mean()


# -------------------------
# Data
# -------------------------
full = Data()

train_idx = np.load("data/train_idx.npy")
val_idx = np.load("data/val_idx.npy")

train_loader = DataLoader(Subset(full, train_idx), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(Subset(full, val_idx), batch_size=BATCH_SIZE)


# -------------------------
# Train
# -------------------------
model = Model().to(DEVICE)
opt = optim.Adam(model.parameters(), lr=LR)

best = float("inf")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        opt.zero_grad()
        out = model(x)

        loss = custom_loss(out, y)
        loss.backward()
        opt.step()

        train_loss += loss.item()

    model.eval()
    val_loss = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            val_loss += custom_loss(model(x), y).item()

    print(f"Epoch {epoch+1}: Train {train_loss:.3f} | Val {val_loss:.3f}")

    if val_loss < best:
        best = val_loss
        torch.save(model.state_dict(), "checkpoints/best_custom.pth")
        print("Saved best custom model")