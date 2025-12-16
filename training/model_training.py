import os
import random
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ======================================================
# CONFIGURATION
# ======================================================

# Only allow real image files (prevents Thumbs.db crash)
VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")

# Training parameters
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 1e-3
EMBEDDING_SIZE = 128

# Image size (must stay consistent everywhere)
IMG_HEIGHT = 155
IMG_WIDTH = 220


# ======================================================
# DATASET: Generates signature pairs
# ======================================================

class SignaturePairsDataset(Dataset):
    """
    Siamese Dataset:
    - Positive pair  -> genuine vs genuine (label = 1)
    - Negative pair  -> genuine vs forged  (label = 0)
    """

    def __init__(self, genuine_dir, forged_dir, transform=None):
        self.transform = transform

        # Load only valid image files (ignores Thumbs.db)
        self.genuine_images = [
            os.path.join(genuine_dir, f)
            for f in os.listdir(genuine_dir)
            if f.lower().endswith(VALID_EXTENSIONS)
        ]

        self.forged_images = [
            os.path.join(forged_dir, f)
            for f in os.listdir(forged_dir)
            if f.lower().endswith(VALID_EXTENSIONS)
        ]

        print(f"[INFO] Genuine images: {len(self.genuine_images)}")
        print(f"[INFO] Forged images : {len(self.forged_images)}")

        # Build positive pairs (genuine-genuine)
        self.pos_pairs = [
            (self.genuine_images[i], self.genuine_images[j], 1)
            for i in range(len(self.genuine_images))
            for j in range(i + 1, len(self.genuine_images))
        ]

        # Build negative pairs (genuine-forged)
        self.neg_pairs = [
            (g, random.choice(self.forged_images), 0)
            for g in self.genuine_images
        ]

        # Combine & shuffle
        self.pairs = self.pos_pairs + self.neg_pairs
        random.shuffle(self.pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_path, img2_path, label = self.pairs[idx]

        # Load images as grayscale
        img1 = Image.open(img1_path).convert("L")
        img2 = Image.open(img2_path).convert("L")

        # Apply transforms
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)


# ======================================================
# SIAMESE CNN ENCODER
# ======================================================

class SiameseCNN(nn.Module):
    """
    CNN that converts a signature image into a feature embedding
    """

    def __init__(self, embedding_size=EMBEDDING_SIZE):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Dynamically calculate FC input size (FIXES SHAPE ERROR)
        self._to_linear = None
        self._get_conv_output()

        self.fc1 = nn.Linear(self._to_linear, 256)
        self.fc2 = nn.Linear(256, embedding_size)

    def _get_conv_output(self):
        """Automatically computes flattened CNN output size"""
        with torch.no_grad():
            x = torch.zeros(1, 1, IMG_HEIGHT, IMG_WIDTH)
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            self._to_linear = x.view(1, -1).size(1)

    def forward_one(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # Normalize embedding (important for similarity)
        return F.normalize(x, p=2, dim=1)

    def forward(self, x1, x2):
        return self.forward_one(x1), self.forward_one(x2)


# ======================================================
# CONTRASTIVE LOSS
# ======================================================

class ContrastiveLoss(nn.Module):
    """
    Loss for Siamese Networks
    """

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, out1, out2, label):
        distance = F.pairwise_distance(out1, out2)
        loss = (
            label * distance.pow(2) +
            (1 - label) * torch.clamp(self.margin - distance, min=0.0).pow(2)
        )
        return loss.mean()


# ======================================================
# TRAINING SCRIPT
# ======================================================

def main():
    # Base directory (training/)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    genuine_dir = os.path.join(BASE_DIR, "datasets", "full_org")
    forged_dir = os.path.join(BASE_DIR, "datasets", "full_forg")

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor()
    ])

    # Dataset & loader
    dataset = SignaturePairsDataset(genuine_dir, forged_dir, transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # Model, loss, optimizer
    model = SiameseCNN().to(device)
    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for img1, img2, labels in loader:
            img1, img2, labels = (
                img1.to(device),
                img2.to(device),
                labels.to(device)
            )

            optimizer.zero_grad()
            out1, out2 = model(img1, img2)
            loss = criterion(out1, out2, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f}")

    # Save trained model
    model_path = os.path.join(BASE_DIR, "signature_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"[SUCCESS] Model saved at: {model_path}")


# ======================================================
# ENTRY POINT
# ======================================================
 
if __name__ == "__main__":
    main()
