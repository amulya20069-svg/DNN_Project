import numpy as np
import torchvision
import torchvision.transforms as transforms

# Load CIFAR-10 test set
transform = transforms.Compose([
    transforms.ToTensor()
])

cifar10_test = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# Load CIFAR-10H soft labels
# Change filename if needed
cifar10h = np.load("./data/cifar10h-probs.npy")

print("CIFAR-10 test images:", len(cifar10_test))
print("CIFAR-10H soft labels shape:", cifar10h.shape)

# Check one sample
image, hard_label = cifar10_test[0]
soft_label = cifar10h[0]

print("Image shape:", image.shape)
print("Hard label:", hard_label)
print("Soft label vector:", soft_label)
print("Soft label length:", len(soft_label))
print("Soft label sum:", soft_label.sum())