import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
import os

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------- CREATE FOLDERS ----------------
os.makedirs("models", exist_ok=True)

# ---------------- TRANSFORMS ----------------
# ResNet18 expects 224x224 input (ImageNet standard)
# Mean/std are ImageNet values — must use these with pretrained ResNet
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.2,
        hue=0.1
    ),
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05),
        scale=(0.95, 1.05)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # ImageNet mean — required for ResNet18
        std=[0.229, 0.224, 0.225]     # ImageNet std  — required for ResNet18
    )
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------- DATASET ----------------
full_dataset = ImageFolder("datasets/stego_dataset")

total = len(full_dataset)
indices = list(range(total))
np.random.seed(42)
np.random.shuffle(indices)

train_size    = int(0.8 * total)
train_indices = indices[:train_size]
test_indices  = indices[train_size:]

class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset   = dataset
        self.indices   = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        if self.transform:
            img = self.transform(img)
        return img, label

full_dataset.transform = None

train_data = TransformSubset(full_dataset, train_indices, transform_train)
test_data  = TransformSubset(full_dataset, test_indices,  transform_test)

print(f"\nDataset : {total} total images")
print(f"Train   : {len(train_data)} images")
print(f"Test    : {len(test_data)} images")
print(f"Classes : {full_dataset.classes}")

train_loader = DataLoader(train_data, batch_size=32, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_data,  batch_size=32, shuffle=False, num_workers=0)

# ---------------- RESNET18 MODEL ----------------
# Load pretrained ResNet18 — already knows edges, textures, patterns
# from 1.2 million ImageNet images
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Freeze all early layers — keep ImageNet knowledge intact
# Only train the last few layers for stego-specific features
for name, param in model.named_parameters():
    param.requires_grad = False

# Unfreeze last 2 residual blocks (layer3, layer4) + final FC
# These will be fine-tuned for stego detection
for name, param in model.named_parameters():
    if any(layer in name for layer in ["layer3", "layer4", "fc"]):
        param.requires_grad = True

# Replace final FC layer for binary classification (cover vs stego)
model.fc = nn.Sequential(
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, 2)
)

model = model.to(device)

# Count trainable parameters
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_p   = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters    : {total_p:,}")
print(f"Trainable parameters: {trainable:,}")
print(f"Frozen parameters   : {total_p - trainable:,}")

# ---------------- OPTIMIZER ----------------
# Only pass trainable parameters to optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.0005,          # lower LR for fine-tuning pretrained model
    weight_decay=1e-4
)

# Reduce LR every 8 epochs
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

# ---------------- TRAINING ----------------
best_acc = 0
EPOCHS   = 25           # fewer epochs needed — ResNet converges faster

print(f"\nTraining ResNet18 for {EPOCHS} epochs...\n")

for epoch in range(EPOCHS):
    model.train()
    total_loss    = 0
    train_correct = 0
    train_total   = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss   += loss.item()
        _, predicted  = torch.max(outputs, 1)
        train_total   += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_acc = 100 * train_correct / train_total

    # Evaluation
    model.eval()
    correct = 0
    total   = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs        = model(images)
            _, predicted   = torch.max(outputs, 1)
            total         += labels.size(0)
            correct       += (predicted == labels).sum().item()

    test_acc   = 100 * correct / total
    current_lr = optimizer.param_groups[0]['lr']

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {total_loss:.2f} | "
          f"Train: {train_acc:.2f}% | Test: {test_acc:.2f}% | LR: {current_lr:.6f}")

    if test_acc > best_acc:
        best_acc = test_acc
        try:
            torch.save(model.state_dict(), "models/best_model.pt")
            print(f"  ✅ Best model saved ({best_acc:.2f}%)")
        except Exception as e:
            print(f"  ❌ Save failed: {e}")

    scheduler.step()

# Save accuracy
with open("models/stego_accuracy.txt", "w") as f:
    f.write(f"{best_acc:.2f}")

print(f"\n🔥 BEST TEST ACCURACY : {best_acc:.2f}%")
print("✅ Model saved to      : models/best_model.pt")
print("✅ Accuracy saved to   : models/stego_accuracy.txt")