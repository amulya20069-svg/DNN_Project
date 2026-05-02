import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


def compute_entropy(prob_vector):
    prob_vector = np.array(prob_vector, dtype=np.float64)
    prob_vector = np.clip(prob_vector, 1e-12, 1.0)
    return -np.sum(prob_vector * np.log2(prob_vector))


class CIFAR10HCustom(Dataset):
    def __init__(self, root="./data"):
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        self.cifar10_test = torchvision.datasets.CIFAR10(
            root=root,
            train=False,
            download=True,
            transform=self.transform
        )

        self.soft_labels = np.load(f"{root}/cifar10h-probs.npy")

        assert len(self.cifar10_test) == len(self.soft_labels), "Image-label count mismatch"
        assert self.soft_labels.shape[1] == 10, "Soft labels must have 10 classes"

        self.entropies = np.array([compute_entropy(p) for p in self.soft_labels])

    def __len__(self):
        return len(self.cifar10_test)

    def __getitem__(self, idx):
        image, hard_label = self.cifar10_test[idx]
        soft_label = torch.tensor(self.soft_labels[idx], dtype=torch.float32)
        entropy = torch.tensor(self.entropies[idx], dtype=torch.float32)
        return image, hard_label, soft_label, entropy


dataset = CIFAR10HCustom(root="./data")

print("Total samples:", len(dataset))

image, hard_label, soft_label, entropy = dataset[0]

print("Image shape:", image.shape)
print("Hard label:", hard_label)
print("Soft label:", soft_label)
print("Entropy:", entropy.item())

all_entropies = dataset.entropies
print("Min entropy:", np.min(all_entropies))
print("Max entropy:", np.max(all_entropies))
print("Mean entropy:", np.mean(all_entropies))

plt.figure(figsize=(8, 5))
plt.hist(all_entropies, bins=30)
plt.xlabel("Entropy")
plt.ylabel("Number of images")
plt.title("Histogram of CIFAR-10H Entropy")
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/entropy_histogram.png")
plt.close()