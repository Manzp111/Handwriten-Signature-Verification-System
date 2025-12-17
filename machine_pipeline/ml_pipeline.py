import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

class SiameseSignatureML:
    """
    ML pipeline class for signature verification.
    - Loads trained Siamese CNN weights
    - Extracts embeddings from signature images
    - Optional: compare embeddings
    """

    class CNNEncoder(nn.Module):
        def __init__(self, embedding_size=128):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)

            # Dynamically compute flattened size
            dummy_input = torch.zeros(1, 1, 155, 220)
            dummy_output = self._forward_conv(dummy_input)
            self.flattened_size = dummy_output.view(1, -1).size(1)

            self.fc1 = nn.Linear(self.flattened_size, 256)
            self.fc2 = nn.Linear(256, embedding_size)

        def _forward_conv(self, x):
            x = F.relu(self.conv1(x))
            x = self.pool(F.relu(self.conv2(x)))
            return x

        def forward(self, x):
            x = self._forward_conv(x)
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return F.normalize(x, p=2, dim=1)

    def __init__(self, model_path="v2/signature_model.pth", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.CNNEncoder().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize((155, 220)),
            transforms.ToTensor()
        ])

    def extract_embedding(self, image_path):
        img = Image.open(image_path).convert("L")
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(img_tensor)
        return embedding.cpu().numpy()[0]

    @staticmethod
    def compare_embeddings(emb1, emb2):
        emb1 = emb1 / np.linalg.norm(emb1)
        emb2 = emb2 / np.linalg.norm(emb2)
        return np.dot(emb1, emb2)
