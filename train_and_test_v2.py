import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

# ===== 1. Data transforms =====
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ===== 2. Load dataset =====
dataset = datasets.ImageFolder(root="data", transform=data_transforms)
print(f"Total images: {len(dataset)}")
print(f"Classes: {dataset.classes}")

# ===== 3. Train-test split =====
indices = list(range(len(dataset)))
labels = [dataset[i][1] for i in indices]
train_idx, test_idx = train_test_split(
    indices, test_size=0.2, stratify=labels, random_state=42
)

train_dataset = Subset(dataset, train_idx)
test_dataset = Subset(dataset, test_idx)

# ===== 4. Handle class imbalance =====
class_counts = np.bincount([dataset[i][1] for i in train_idx])
class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
sample_weights = [class_weights[dataset[i][1]] for i in train_idx]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ===== 5. Model setup =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet34(weights="IMAGENET1K_V1")

# Unfreeze last few layers for fine-tuning
for name, param in model.named_parameters():
    if "layer4" in name or "fc" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model = model.to(device)

# ===== 6. Loss, optimizer, scheduler =====
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# ===== 7. Training loop =====
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    scheduler.step()
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")

    # Save checkpoint
    torch.save(model.state_dict(), f"checkpoint_epoch_{epoch+1}.pth")
    print(f"✅ Checkpoint saved for epoch {epoch+1}")

# ===== 8. Evaluation =====
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

acc = accuracy_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)
report = classification_report(all_labels, all_preds, target_names=dataset.classes)

print(f"\n✅ Test Accuracy: {acc*100:.2f}%")
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", report)

torch.save(model.state_dict(), "cancer_model_v2.pth")
print("✅ Model saved as cancer_model_v2.pth")
