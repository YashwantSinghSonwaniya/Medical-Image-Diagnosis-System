import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

train_dir = os.path.join(BASE_DIR, "chest_xray/train")
val_dir = os.path.join(BASE_DIR, "chest_xray/val")

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Datasets
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)

# Loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Model
model = get_model(num_classes=len(train_dataset.classes))
model.to(device)

# Loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 5

for epoch in range(epochs):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images, labels = images.to(device), labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    print(f"Epoch [{epoch+1}/{epochs}]")
    print(f"Loss: {running_loss:.4f}")
    print(f"Training Accuracy: {accuracy:.2f}%")

# Save model
models_dir = os.path.join(BASE_DIR, "models")

os.makedirs(models_dir, exist_ok=True)

torch.save(model.state_dict(),
           os.path.join(models_dir, "model.pth"))

print("Model saved successfully!")





### first code

# import torch
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# from model import get_model

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
# ])

# dataset = datasets.ImageFolder("dataset/", transform=transform)
# loader = DataLoader(dataset, batch_size=16, shuffle=True)

# model = get_model(num_classes=len(dataset.classes))
# model.to(device)

# criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# for epoch in range(5):
#     for images, labels in loader:
#         images, labels = images.to(device), labels.to(device)

#         outputs = model(images)
#         loss = criterion(outputs, labels)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     print(f"Epoch {epoch} Loss: {loss.item()}")

# torch.save(model.state_dict(), "models/model.pth")