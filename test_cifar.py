import torch
import torchvision
import torchvision.transforms as transforms

# Transform (convert image to tensor)
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load CIFAR-10 training data
trainset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=4,
    shuffle=True
)

# Load test data
testset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=4,
    shuffle=False
)

print("Train size:", len(trainset))
print("Test size:", len(testset))

# Show one sample
images, labels = next(iter(trainloader))
print("Sample image shape:", images[0].shape)
print("Sample label:", labels[0].item())