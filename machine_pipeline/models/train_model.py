import os
import shutil
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from collections import defaultdict





# 1. SYSTEM CONFIGURATION
# the model is trained wit help of google colab
DRIVE_PATH = "/content/drive/MyDrive/Machine Learning" 
DRIVE_GEN_DIR = f"{DRIVE_PATH}/datasets/full_org"
DRIVE_FORG_DIR = f"{DRIVE_PATH}/datasets/full_forg"

LOCAL_DATA_DIR = "/content/local_dataset"
LOCAL_GEN_DIR = os.path.join(LOCAL_DATA_DIR, "full_org")
LOCAL_FORG_DIR = os.path.join(LOCAL_DATA_DIR, "full_forg")


FINAL_MODEL_SAVE_PATH = f"{DRIVE_PATH}/signature_verifier_v2_final.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Evaluation/Training Params
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-4
EMBEDDING_DIM = 128
SECURITY_THRESHOLD = 0.4  # Decision boundary for Match vs No-Match

# 2. DATA ENGINEERING & CLEANING

def sync_data():
    if not os.path.exists(LOCAL_DATA_DIR):
        os.makedirs(LOCAL_DATA_DIR)
        print("Syncing data to Local SSD...")
        os.system(f'cp -r "{DRIVE_GEN_DIR}" "{LOCAL_GEN_DIR}"')
        os.system(f'cp -r "{DRIVE_FORG_DIR}" "{LOCAL_FORG_DIR}"')
        print("Data sync complete.")

class SignatureProcessor:
    """Removes background noise to focus only on handwriting geometry."""
    def __call__(self, img):
        img = ImageOps.grayscale(img)
        img = ImageOps.autocontrast(img)
        # Binarization (Pure Black/White)
        img = img.point(lambda p: 255 if p > 128 else 0)
        return img.convert("RGB")

# 3.SIAMESE ARCHITECTURE

class SignatureFeatureEncoder(nn.Module):
    def __init__(self, dim=EMBEDDING_DIM):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, dim)
        )

    def forward_single(self, x):
        return F.normalize(self.backbone(x), p=2, dim=1)

    def forward(self, x1, x2):
        return self.forward_single(x1), self.forward_single(x2)

class DistanceLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, o1, o2, y):
        dist = F.pairwise_distance(o1, o2)
        loss = y * dist.pow(2) + (1-y) * torch.clamp(self.margin - dist, min=0.0).pow(2)
        return loss.mean()

# 4. DATASET & EVALUATION LOGIC

class SignatureDataset(Dataset):
    def __init__(self, gen_dir, forg_dir, transform=None):
        self.transform = transform
        self.pairs = []
        gen_map, forg_map = defaultdict(list), defaultdict(list)

        for folder, d, prefix in [(gen_dir, gen_map, "original_"), (forg_dir, forg_map, "forgeries_")]:
            for f in sorted(os.listdir(folder)):
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    writer = f.split('_')[1]
                    d[writer].append(os.path.join(folder, f))

        for w in gen_map.keys():
            g, f = gen_map[w], forg_map[w]
            for i in range(len(g)-1): self.pairs.append((g[i], g[i+1], 1)) # Genuine
            for i in range(min(len(g), len(f))): self.pairs.append((g[i], f[i], 0)) # Forgery
        
        random.shuffle(self.pairs)

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        p1, p2, label = self.pairs[idx]
        img1, img2 = Image.open(p1), Image.open(p2)
        if self.transform: img1, img2 = self.transform(img1), self.transform(img2)
        return img1, img2, torch.tensor(label, dtype=torch.float32)

def perform_full_analysis(model, loader, threshold):
    model.eval()
    y_true, y_pred, distances = [], [], []
    
    with torch.no_grad():
        for i1, i2, label in loader:
            o1, o2 = model(i1.to(DEVICE), i2.to(DEVICE))
            d = F.pairwise_distance(o1, o2)
            y_true.extend(label.numpy())
            y_pred.extend((d < threshold).float().cpu().numpy())
            distances.extend(d.cpu().numpy())

    # Text Analysis
    print("\n" + "="*40)
    print("FINAL EVALUATION REPORT")
    print("="*40)
    print(f"Overall Accuracy: {accuracy_score(y_true, y_pred):.2%}")
    print("\nDetailed Metrics:")
    print(classification_report(y_true, y_pred, target_names=["Forgery", "Genuine"]))
    
    # Visualization Analysis
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[0])
    ax[0].set_title("Confusion Matrix (Security Report)")
    ax[0].set_xlabel("Predicted Label")
    ax[0].set_ylabel("Actual Label")

    # 2. Distance Distribution Graph
    dist_gen = [distances[i] for i in range(len(distances)) if y_true[i] == 1]
    dist_forg = [distances[i] for i in range(len(distances)) if y_true[i] == 0]
    sns.kdeplot(dist_gen, fill=True, label="Genuine Pairs", ax=ax[1])
    sns.kdeplot(dist_forg, fill=True, label="Forgery Pairs", ax=ax[1])
    ax[1].axvline(threshold, color='red', linestyle='--', label=f'Threshold ({threshold})')
    ax[1].set_title("How the AI 'Sees' Distance")
    ax[1].legend()

    plt.show()

# 5. MAIN EXECUTION

def main():
    from google.colab import drive
    drive.mount('/content/drive')
    sync_data()

    pipeline = transforms.Compose([
        SignatureProcessor(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_ds = SignatureDataset(LOCAL_GEN_DIR, LOCAL_FORG_DIR, pipeline)
    train_num = int(len(full_ds) * 0.8)
    train_ds, test_ds = random_split(full_ds, [train_num, len(full_ds) - train_num])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = SignatureFeatureEncoder().to(DEVICE)
    criterion = DistanceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(" Starting Training Phase...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for i1, i2, label in train_loader:
            i1, i2, label = i1.to(DEVICE), i2.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()
            o1, o2 = model(i1, i2)
            loss = criterion(o1, o2, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS} | Avg Loss: {total_loss/len(train_loader):.4f}")

    # Final Analysis
    perform_full_analysis(model, test_loader, SECURITY_THRESHOLD)
    
    # Save for later use in Inference
    torch.save(model.state_dict(), FINAL_MODEL_SAVE_PATH)
    print(f"\nSYSTEM READY. Model saved to: {FINAL_MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()