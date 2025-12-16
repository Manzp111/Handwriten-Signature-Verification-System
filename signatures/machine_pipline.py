import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F


# Dataset class for Siamese training

class SignaturePairsDataset(Dataset):
    """
    Generates pairs of images for Siamese CNN:
    - Positive pairs: genuine-genuine (label=1)
    - Negative pairs: genuine-forged (label=0)
    """
    def __init__(self, genuine_dir, forged_dir, transform=None):
        self.genuine_dir = genuine_dir
        self.forged_dir = forged_dir
        self.transform = transform

        self.genuine_images = [os.path.join(genuine_dir, f) for f in os.listdir(genuine_dir)]
        self.forged_images = [os.path.join(forged_dir, f) for f in os.listdir(forged_dir)]

        # Positive pairs
        self.pos_pairs = [(self.genuine_images[i], self.genuine_images[j], 1)
                          for i in range(len(self.genuine_images))
                          for j in range(i+1, len(self.genuine_images))]
        # Negative pairs (one forged per genuine)
        self.neg_pairs = [(g, random.choice(self.forged_images), 0) for g in self.genuine_images]

        self.pairs = self.pos_pairs + self.neg_pairs
        random.shuffle(self.pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_path, img2_path, label = self.pairs[idx]
        img1 = Image.open(img1_path).convert("L")
        img2 = Image.open(img2_path).convert("L")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor([label], dtype=torch.float32)

# ----------------------------
# CNN Encoder for Siamese
# ----------------------------
class SiameseCNN(nn.Module):
    def __init__(self, embedding_size=128):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*38*55, 256)
        self.fc2 = nn.Linear(256, embedding_size)

    def forward_one(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.normalize(x, p=2, dim=1)

    def forward(self, x1, x2):
        return self.forward_one(x1), self.forward_one(x2)

# ----------------------------
# Contrastive Loss
# ----------------------------
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, out1, out2, label):
        dist = F.pairwise_distance(out1, out2)
        loss = label * dist**2 + (1 - label) * torch.clamp(self.margin - dist, min=0.0)**2
        return loss.mean()

# ----------------------------
# Training script
# ----------------------------
from torchvision import transforms

# Image transformations
transform = transforms.Compose([
    transforms.Resize((155, 220)),
    transforms.ToTensor()
])

# Dataset & DataLoader
genuine_dir = "datasets/full_org"
forged_dir = "datasets/full_forg"
dataset = SignaturePairsDataset(genuine_dir, forged_dir, transform=transform)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Model, loss, optimizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SiameseCNN().to(device)
criterion = ContrastiveLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
epochs = 10
for epoch in range(epochs):
    total_loss = 0
    for img1, img2, labels in loader:
        img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
        optimizer.zero_grad()
        out1, out2 = model(img1, img2)
        loss = criterion(out1, out2, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

# Save trained model
torch.save(model.state_dict(), "signature_model.pth")
print("Training complete. Model saved as signature_model.pth")
