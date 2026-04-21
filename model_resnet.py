import torch
import torch.nn as nn
import torchvision.models as models


class CIFAR10HResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.model = models.resnet18(weights=None)

        # Adapt for CIFAR-10 (32x32 images)
        self.model.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        # Remove early maxpool for small images
        self.model.maxpool = nn.Identity()

        # Final layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    model = CIFAR10HResNet18()

    x = torch.randn(8, 3, 32, 32)
    y = model(x)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Output shape:", y.shape)
    print("Total parameters:", total_params)
    print("Trainable parameters:", trainable_params)