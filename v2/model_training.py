import os
import random
from collections import defaultdict
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ======================================================
# CONFIGURATION
# ======================================================
VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 1e-3
EMBEDDING_SIZE = 128
IMG_HEIGHT = 155
IMG_WIDTH = 220
MARGIN = 1.0
CHECKPOINT_PATH = "checkpoint.pth"
MODEL_PATH = "signature_model.pth"

# ======================================================
# DATASET
# ======================================================

class SignatureDataset(Dataset):
    """
    Generates Siamese pairs with:
    - Positive: Genuine-Genuine (same writer)
    - Negative: Genuine-Forged + Genuine-Genuine (different writer)
    """
    def __init__(self, genuine_dir, forged_dir, transform=None):
        self.transform = transform

        self.genuine_by_writer = defaultdict(list)
        self.forged_by_writer = defaultdict(list)

        # Parse genuine images
        for f in os.listdir(genuine_dir):
            if f.lower().endswith(VALID_EXTENSIONS):
                writer = f.split("_")[0]
                self.genuine_by_writer[writer].append(os.path.join(genuine_dir, f))

        # Parse forged images
        for f in os.listdir(forged_dir):
            if f.lower().endswith(VALID_EXTENSIONS):
                writer = f.split("_")[1]  # forgeries_7_1.png -> writer 7
                self.forged_by_writer[writer].append(os.path.join(forged_dir, f))

        self.all_writers = list(self.genuine_by_writer.keys())
        self.pairs = []

        # Build pairs
        for writer in self.all_writers:
            # Positive pairs (same writer)
            images = self.genuine_by_writer[writer]
            for i in range(len(images)):
                for j in range(i+1, len(images)):
                    self.pairs.append((images[i], images[j], 1))

            # Impostor negative pairs (different writers)
            other_writers = [w for w in self.all_writers if w != writer]
            for other_writer in other_writers:
                img1 = random.choice(images)
                img2 = random.choice(self.genuine_by_writer[other_writer])
                self.pairs.append((img1, img2, 0))

            # Genuine vs Forged
            for img in images:
                if writer in self.forged_by_writer:
                    forged_img = random.choice(self.forged_by_writer[writer])
                    self.pairs.append((img, forged_img, 0))

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

        return img1, img2, torch.tensor(label, dtype=torch.float32)

# ======================================================
# SIAMESE CNN
# ======================================================

class SiameseCNN(nn.Module):
    def __init__(self, embedding_size=EMBEDDING_SIZE):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self._to_linear = None
        self._get_conv_output()
        self.fc1 = nn.Linear(self._to_linear, 256)
        self.fc2 = nn.Linear(256, embedding_size)

    def _get_conv_output(self):
        with torch.no_grad():
            x = torch.zeros(1,1,IMG_HEIGHT,IMG_WIDTH)
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            self._to_linear = x.view(1,-1).size(1)

    def forward_one(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.normalize(x, p=2, dim=1)

    def forward(self, x1, x2):
        return self.forward_one(x1), self.forward_one(x2)

# ======================================================
# CONTRASTIVE LOSS
# ======================================================

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=MARGIN):
        super().__init__()
        self.margin = margin

    def forward(self, out1, out2, label):
        distance = F.pairwise_distance(out1, out2)
        loss = label*distance.pow(2) + (1-label)*torch.clamp(self.margin-distance, min=0.0).pow(2)
        return loss.mean()

# ======================================================
# TRAINING
# ======================================================

def train_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    genuine_dir = os.path.join(BASE_DIR, "datasets", "full_org")
    forged_dir = os.path.join(BASE_DIR, "datasets", "full_forg")

    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor()
    ])

    dataset = SignatureDataset(genuine_dir, forged_dir, transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    model = SiameseCNN().to(device)
    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Resume from checkpoint if exists
    start_epoch = 0
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"[INFO] Resuming training from epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0
        all_labels = []
        all_preds = []

        for img1, img2, labels in loader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

            optimizer.zero_grad()
            out1, out2 = model(img1, img2)
            loss = criterion(out1, out2, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Collect predictions for metrics
            dist = F.pairwise_distance(out1, out2)
            pred = (dist < 0.5).float()
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(loader)
        acc = accuracy_score(all_labels, all_preds)
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f} - Accuracy: {acc:.4f}")

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict()
        }, CHECKPOINT_PATH)

    # Save final model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"[SUCCESS] Model saved at: {MODEL_PATH}")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", classification_report(all_labels, all_preds))

    # Optional: Plot confusion matrix
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

# ======================================================
# ENTRY POINT
# ======================================================

if __name__ == "__main__":
    train_model()
