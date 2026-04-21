import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset


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


# transforms
transform = transforms.Compose([
    transforms.ToTensor()
])

# full dataset
full_dataset = CIFAR10HCustom(root="./data", transform=transform)

# load saved split indices
train_idx = np.load("data/train_idx.npy")
val_idx = np.load("data/val_idx.npy")
test_idx = np.load("data/test_idx.npy")

# subsets
train_dataset = Subset(full_dataset, train_idx)
val_dataset = Subset(full_dataset, val_idx)
test_dataset = Subset(full_dataset, test_idx)

# dataloaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# check
print("Train batches:", len(train_loader))
print("Val batches:", len(val_loader))
print("Test batches:", len(test_loader))

# inspect one batch
images, hard_labels, soft_labels, entropies = next(iter(train_loader))

print("Image batch shape:", images.shape)
print("Hard label batch shape:", hard_labels.shape)
print("Soft label batch shape:", soft_labels.shape)
print("Entropy batch shape:", entropies.shape)

print("First soft label:", soft_labels[0])
print("First entropy:", entropies[0].item())