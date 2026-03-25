import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import glob

# 1. Custom CNN for Pothole Severity (M2)
# Using EfficientNet-V2-S as it is lightweight and excellent at texture extraction
class SeverityCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(SeverityCNN, self).__init__()
        # Load pre-trained EfficientNetV2
        self.base_model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        
        # Modify the final layer for our 3 classes (Minor, Medium, Major)
        num_ftrs = self.base_model.classifier[1].in_features
        self.base_model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)

# 2. Dataset Loader with Contextual Cropping Logic
class PotholeSeverityDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Assuming folder structure: root/minor/*.jpg, root/medium/*.jpg, root/major/*.jpg
        self.class_map = {"minor": 0, "medium": 1, "major": 2}
        
        for label_name, label_idx in self.class_map.items():
            folder = os.path.join(root_dir, label_name)
            if not os.path.exists(folder): continue
            for img_path in glob.glob(os.path.join(folder, "*.jpg")):
                self.image_paths.append(img_path)
                self.labels.append(label_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

def train_m2():
    # Hyperparameters
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    EPOCHS = 20
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data Augmentation (Crucial for M2)
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Potholes look different in sun/shade
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Initialize Dataset
    dataset = PotholeSeverityDataset(root_dir="severity_dataset/train", transform=data_transforms)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize Model, Loss, Optimizer
    model = SeverityCNN(num_classes=3).to(DEVICE)
    criterion = nn.CrossEntropyLoss() # Standard classification loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print(f"Starting M2 Training on {DEVICE}...")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch+1}/{EPOCHS} Loss: {epoch_loss:.4f}")

    # Save the custom model
    os.makedirs("weights", exist_ok=True)
    torch.save(model.state_dict(), "weights/m2_severity_custom.pth")
    print("M2 Training Complete. Model saved to weights/m2_severity_custom.pth")

if __name__ == "__main__":
    train_m2()
