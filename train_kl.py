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
BATCH_SIZE = 64
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
PATIENCE = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
def compute_entropy(prob_vector):
    prob_vector = np.array(prob_vector, dtype=np.float64)
    prob_vector = np.clip(prob_vector, 1e-12, 1.0)
    return -np.sum(prob_vector * np.log2(prob_vector))
class CIFAR10HCustom(Dataset):
    def __init__(self, root="./data", transform=None):
        self.transform = transform

        self.cifar10_test = torchvision.datasets.CIFAR10(
            root=root,
            train=False,
            download=True,
            transform=self.transform
        )

        self.soft_labels = np.load(f"{root}/cifar10h-probs.npy")
        self.entropies = np.array([compute_entropy(p) for p in self.soft_labels])

        assert len(self.cifar10_test) == len(self.soft_labels), "Mismatch between images and soft labels"
        assert self.soft_labels.shape[1] == 10, "Soft labels must have shape (N, 10)"

    def __len__(self):
        return len(self.cifar10_test)

    def __getitem__(self, idx):
        image, hard_label = self.cifar10_test[idx]
        soft_label = torch.tensor(self.soft_labels[idx], dtype=torch.float32)
        entropy = torch.tensor(self.entropies[idx], dtype=torch.float32)
        hard_label = torch.tensor(hard_label, dtype=torch.long)

        return image, hard_label, soft_label, entropy
class CIFAR10HResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.model = models.resnet18(weights=None)

        # Adapt for CIFAR-10 (32x32)
        self.model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.model.maxpool = nn.Identity()

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        logits = self.model(x)
        return logits   # return logits, NOT softmax
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.ToTensor()
])

full_dataset_train = CIFAR10HCustom(root="./data", transform=train_transform)
full_dataset_val = CIFAR10HCustom(root="./data", transform=val_transform)

train_idx = np.load("data/train_idx.npy")
val_idx = np.load("data/val_idx.npy")

train_dataset = Subset(full_dataset_train, train_idx)
val_dataset = Subset(full_dataset_val, val_idx)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
model = CIFAR10HResNet18().to(DEVICE)

# KL divergence expects log-probabilities as input
criterion = nn.KLDivLoss(reduction="batchmean")

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for images, hard_labels, soft_labels, entropies in loader:
        images = images.to(device)
        soft_labels = soft_labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        log_probs = torch.log_softmax(logits, dim=1)

        loss = criterion(log_probs, soft_labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for images, hard_labels, soft_labels, entropies in loader:
            images = images.to(device)
            soft_labels = soft_labels.to(device)

            logits = model(images)
            log_probs = torch.log_softmax(logits, dim=1)

            loss = criterion(log_probs, soft_labels)
            running_loss += loss.item()

    return running_loss / len(loader)
train_losses = []
val_losses = []

best_val_loss = float("inf")
best_model_wts = copy.deepcopy(model.state_dict())
epochs_without_improvement = 0

print("Training on device:", DEVICE)

for epoch in range(NUM_EPOCHS):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
    val_loss = validate_one_epoch(model, val_loader, criterion, DEVICE)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(best_model_wts, "checkpoints/best_resnet18_kl.pth")
        print("Best model saved.")
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print("Early stopping triggered.")
        break
model.load_state_dict(best_model_wts)
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("KL Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/kl_loss_curve.png")
plt.show()

print("Training complete.")
print("Best validation loss:", best_val_loss)
print("Best model path: checkpoints/best_resnet18_kl.pth")
print("Loss curve saved to: outputs/kl_loss_curve.png")