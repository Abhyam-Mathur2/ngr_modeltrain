import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import glob
import numpy as np

# 1. Research-Level Dual-Input CNN
# This model combines Image Features (CNN) with Geometric Features (Metadata)
class ResearchSeverityModel(nn.Module):
    def __init__(self, num_classes=3, meta_dim=3):
        super(ResearchSeverityModel, self).__init__()
        # Stream 1: Image Features (EfficientNetV2)
        self.cnn_base = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        cnn_out_features = self.cnn_base.classifier[1].in_features
        self.cnn_base.classifier = nn.Identity() # Remove final layer to get features
        
        # Stream 2: Metadata Features (MLP)
        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU()
        )
        
        # Fusion Layer
        self.classifier = nn.Sequential(
            nn.Linear(cnn_out_features + 16, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, image, metadata):
        img_feats = self.cnn_base(image)
        meta_feats = self.meta_mlp(metadata)
        combined = torch.cat((img_feats, meta_feats), dim=1)
        return self.classifier(combined)

# 2. Advanced Dataset with Metadata extraction
class PotholeResearchDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        
        self.class_map = {"minor": 0, "medium": 1, "major": 2}
        
        for label_name, label_idx in self.class_map.items():
            folder = os.path.join(root_dir, label_name)
            if not os.path.exists(folder): continue
            for img_path in glob.glob(os.path.join(folder, "*.jpg")):
                # In a real research scenario, we would parse the metadata from 
                # a companion .txt file or image filename.
                # For this prototype, we simulate geometric metadata: [Area, AspectRatio, Confidence]
                # In main.py, you will pass real values from M1.
                img = Image.open(img_path)
                w, h = img.size
                meta = [w * h / (640*640), w/h, 0.9] # Normalized Area, AspectRatio, Dummy Conf
                
                self.data.append({
                    "path": img_path,
                    "label": label_idx,
                    "meta": torch.tensor(meta, dtype=torch.float32)
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = Image.open(item["path"]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, item["meta"], item["label"]

def train_m2_research():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = PotholeResearchDataset(root_dir="../severity_dataset/train", transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = ResearchSeverityModel(num_classes=3).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print(f"Starting Research-Level M2 Training (Dual-Stream) on {DEVICE}...")
    
    for epoch in range(5): # Short run for demonstration
        model.train()
        for images, metas, labels in loader:
            images, metas, labels = images.to(DEVICE), metas.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images, metas)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1} Complete.")

    torch.save(model.state_dict(), "weights/m2_research_dual_stream.pth")
    print("M2 Research Model saved.")

if __name__ == "__main__":
    train_m2_research()
