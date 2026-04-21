import numpy as np

# total samples
N = 10000

# fixed seed (IMPORTANT for reproducibility)
np.random.seed(42)

# shuffle indices
indices = np.arange(N)
np.random.shuffle(indices)

# split
train_idx = indices[:6000]
val_idx = indices[6000:8000]
test_idx = indices[8000:]

print("Train:", len(train_idx))
print("Val:", len(val_idx))
print("Test:", len(test_idx))

# save indices (VERY IMPORTANT for project reproducibility)
np.save("data/train_idx.npy", train_idx)
np.save("data/val_idx.npy", val_idx)
np.save("data/test_idx.npy", test_idx)

print("Splits saved!")