from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# ----------------------------
# Preprocess signature image
# ----------------------------
def preprocess_image(file):
    image = Image.open(file).convert("L")
    transform = transforms.Compose([
        transforms.Resize((155, 220)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    return transform(image).unsqueeze(0)

# Siamese Network Model

class SiameseCNN(nn.Module):
    def __init__(self, embedding_dim=128):
        super(SiameseCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*16*24, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )

    def forward_once(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        return out1, out2

# ----------------------------
# Cosine similarity
# ----------------------------
def cosine_similarity(emb1, emb2):
    emb1 = torch.tensor(emb1)
    emb2 = torch.tensor(emb2)
    return F.cosine_similarity(emb1, emb2.unsqueeze(0)).item()

# ----------------------------
# Signature Verification
# ----------------------------
def verify_signature(user, file, threshold=0.95, model=None, device=None):
    input_tensor = preprocess_image(file).to(device).flatten()
    best_score = 0

    for sig in user.signatures.all():
        stored_embedding = torch.tensor(sig.features)
        score = cosine_similarity(input_tensor, stored_embedding)
        if score > best_score:
            best_score = score

    result = "Accepted" if best_score >= threshold else "Rejected"
    return {"result": result, "score": best_score}
