import os
import copy
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.models as models


# SETTINGS
BATCH_SIZE = 128
NUM_EPOCHS = 5
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("checkpoints", exist_ok=True)


# DATASET
class CIFAR10HCustom(Dataset):
    def __init__(self, root="./data", transform=None):
        self.transform = transform

        self.cifar10 = torchvision.datasets.CIFAR10(
            root=root, train=False, download=True, transform=self.transform
        )

        self.soft_labels = np.load(f"{root}/cifar10h-probs.npy")

    def __len__(self):
        return len(self.cifar10)

    def __getitem__(self, idx):
        img, _ = self.cifar10[idx]
        soft = torch.tensor(self.soft_labels[idx], dtype=torch.float32)
        return img, soft


# MODEL
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.resnet18(weights=None)
        self.net.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.net.maxpool = nn.Identity()
        self.net.fc = nn.Linear(self.net.fc.in_features, 10)

    def forward(self, x):
        return self.net(x)


# DATA
transform = transforms.Compose([transforms.ToTensor()])

full = CIFAR10HCustom(transform=transform)

train_idx = np.load("data/train_idx.npy")
val_idx = np.load("data/val_idx.npy")

train_loader = DataLoader(Subset(full, train_idx), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(Subset(full, val_idx), batch_size=BATCH_SIZE)


# LOSS (soft cross entropy)
def soft_cross_entropy(pred, target):
    log_probs = torch.log_softmax(pred, dim=1)
    return -(target * log_probs).sum(dim=1).mean()


# TRAIN
model = Model().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)

best_val = float("inf")

for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        out = model(x)

        loss = soft_cross_entropy(out, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    val_loss = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            val_loss += soft_cross_entropy(out, y).item()

    print(f"Epoch {epoch+1}: Train {train_loss:.3f} | Val {val_loss:.3f}")

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), "checkpoints/best_ce.pth")
        print("Saved best CE model")