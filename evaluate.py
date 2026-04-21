import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.models as models
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr, spearmanr
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def compute_entropy(p):
    p = np.clip(p, 1e-12, 1.0)
    return -np.sum(p * np.log2(p))

class CIFAR10HCustom(Dataset):
    def __init__(self, root="./data"):
        transform = transforms.Compose([transforms.ToTensor()])

        self.cifar10_test = torchvision.datasets.CIFAR10(
            root=root, train=False, download=True, transform=transform
        )

        self.soft_labels = np.load(f"{root}/cifar10h-probs.npy")
        self.entropies = np.array([compute_entropy(p) for p in self.soft_labels])

    def __len__(self):
        return len(self.cifar10_test)

    def __getitem__(self, idx):
        image, _ = self.cifar10_test[idx]
        soft_label = torch.tensor(self.soft_labels[idx], dtype=torch.float32)
        entropy = self.entropies[idx]
        return image, soft_label, entropy
class CIFAR10HResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.resnet18(weights=None)

        self.net.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.net.maxpool = nn.Identity()
        self.net.fc = nn.Linear(self.net.fc.in_features, 10)

    def forward(self, x):
        return self.net(x)

dataset = CIFAR10HCustom()

test_idx = np.load("data/test_idx.npy")
test_dataset = Subset(dataset, test_idx)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
model = CIFAR10HResNet18().to(DEVICE)
model.load_state_dict(torch.load("checkpoints/best_custom.pth"))
model.eval()
kl_losses = []
jsd_list = []
cosine_list = []
true_entropy = []
pred_entropy = []

criterion = nn.KLDivLoss(reduction="batchmean")


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


with torch.no_grad():
    for images, soft_labels, entropies in test_loader:
        images = images.to(DEVICE)

        logits = model(images)
        probs = torch.softmax(logits, dim=1)

        for i in range(len(images)):
            p = soft_labels[i].numpy()
            q = probs[i].cpu().numpy()

            # KL
            kl = np.sum(p * np.log((p + 1e-12) / (q + 1e-12)))
            kl_losses.append(kl)

            # JSD
            jsd = jensenshannon(p, q) ** 2
            jsd_list.append(jsd)

            # Cosine
            cosine_list.append(cosine_similarity(p, q))

            # Entropy
            true_entropy.append(entropies[i])
            pred_entropy.append(compute_entropy(q))
print("\n===== FINAL RESULTS =====")
print("KL Divergence:", np.mean(kl_losses))
print("JSD:", np.mean(jsd_list))
print("Cosine Similarity:", np.mean(cosine_list))

pearson_corr, _ = pearsonr(true_entropy, pred_entropy)
spearman_corr, _ = spearmanr(true_entropy, pred_entropy)

print("Pearson Correlation:", pearson_corr)
print("Spearman Correlation:", spearman_corr)