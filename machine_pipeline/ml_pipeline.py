import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image, ImageOps
import numpy as np

# ======================================================
# 1. IMAGE PREPROCESSING
# ======================================================
class SignatureProcessor:
    def __call__(self, img):
        img = ImageOps.grayscale(img)
        img = ImageOps.autocontrast(img)
        img = img.point(lambda p: 255 if p > 128 else 0)
        return img.convert("RGB")

# ======================================================
# 2. FIXED MODEL ARCHITECTURE
# ======================================================
class SiameseCNN(nn.Module):
    def __init__(self, embedding_size=128):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        
        # Based on your error log:
        # fc.0: Linear(512, 512)
        # fc.1: BatchNorm1d(512)
        # fc.3: Linear(512, 128)
        self.backbone.fc = nn.Sequential(
            nn.Linear(512, 512),      # Matches your size mismatch error
            nn.BatchNorm1d(512),     # Matches backbone.fc.1.weight/bias keys
            nn.ReLU(),
            nn.Linear(512, embedding_size) # Matches backbone.fc.3.weight keys
        )

    def forward_one(self, x):
        x = self.backbone(x)
        return F.normalize(x, p=2, dim=1)

# ======================================================
# 3. DJANGO WRAPPER CLASS
# ======================================================
class SiameseSignatureML:
    def __init__(self, model_path):
        self.device = torch.device("cpu")
        self.model = SiameseCNN()
        
        try:
            # Load weights from your .pth file
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            
            # CRITICAL: This fixes the "Expected more than 1 value per channel" error
            # It tells BatchNorm to use the saved average instead of trying to calculate a new one
            self.model.eval() 
            
            print(f"✅ Success: Model architecture synced and set to evaluation mode.")
        except Exception as e:
            print(f"❌ Error loading model: {e}")

        self.transform = transforms.Compose([
            SignatureProcessor(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    def extract_embedding(self, image_input):
        if isinstance(image_input, str):
            img = Image.open(image_input)
        else:
            img = Image.open(image_input).convert("RGB")

        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        # Ensure no gradients are tracked for better performance
        with torch.no_grad():
            embedding = self.model.forward_one(img_tensor)
        
        return embedding.squeeze().numpy()

    @staticmethod
    def calculate_distance(embedding1, embedding2):
        return np.linalg.norm(np.array(embedding1) - np.array(embedding2))