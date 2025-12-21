import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image, ImageOps
import numpy as np
import os
from pathlib import Path

# --- 1. IMAGE PREPROCESSING ---
class SignatureProcessor:
    """Standardizes signature images by removing color and enhancing contrast."""
    def __call__(self, img):
        img = ImageOps.grayscale(img)
        img = ImageOps.autocontrast(img)
        # Binarize: Thresholding to separate ink from paper
        img = img.point(lambda p: 255 if p > 128 else 0)
        return img.convert("RGB")

# --- 2. MODEL ARCHITECTURE ---
class SiameseCNN(nn.Module):
    def __init__(self, embedding_size=128):
        super().__init__()
        # Load ResNet18 backbone
        self.backbone = models.resnet18(weights=None)
        
        # Redefine FC layer to match your trained weights
        # fc.0: Linear, fc.1: BatchNorm, fc.3: Linear
        self.backbone.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, embedding_size)
        )

    def forward_one(self, x):
        x = self.backbone(x)
        # Normalize vectors to a unit sphere for stable distance calculation
        return F.normalize(x, p=2, dim=1)

# --- 3. ML WRAPPER FOR DJANGO ---
class SiameseSignatureML:
    def __init__(self, model_path):
        self.device = torch.device("cpu")
        self.model = SiameseCNN()
        
        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                
                # CRITICAL FIX: Set to eval mode to handle BatchNorm with batch size 1
                self.model.eval() 
                print(f"✅ [SignaSure AI] Model loaded successfully from {model_path}")
            except Exception as e:
                print(f"❌ [SignaSure AI] Error loading weights: {e}")
        else:
            print(f"⚠️ [SignaSure AI] Warning: Model file not found at {model_path}")

        # Image Transformation Pipeline
        self.transform = transforms.Compose([
            SignatureProcessor(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    def extract_embedding(self, image_input):
        """Processes an image and returns a 128-D numerical fingerprint."""
        try:
            # Handle both file paths and Django UploadedFile objects
            img = Image.open(image_input).convert("RGB")
            
            # Prepare tensor and add batch dimension (1, 3, 224, 224)
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)

            # Inference Mode is faster and uses less memory than no_grad()
            with torch.inference_mode():
                self.model.eval() # Ensure eval mode is active
                embedding = self.model.forward_one(img_tensor)
            
            return embedding.squeeze().cpu().numpy()
        except Exception as e:
            print(f"Error during embedding extraction: {e}")
            return None

    @staticmethod
    def calculate_distance(embedding1, embedding2):
        """Calculates Euclidean distance between two signature fingerprints."""
        return np.linalg.norm(np.array(embedding1) - np.array(embedding2))

# --- 4. GLOBAL INITIALIZATION ---
# This ensures the model is loaded once when Django starts.

# Get the project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Set the path to the weights inside the 'training' folder
# Change 'training' to 'machine_pipeline/models' if you moved the file there
MODEL_PATH = os.path.join(BASE_DIR, 'training', 'signature_model.pth')

ml_model = SiameseSignatureML(MODEL_PATH)