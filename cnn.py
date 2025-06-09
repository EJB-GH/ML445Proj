import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define image transforms
my_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Load dataset
path = 'Alzheimer_MRI_4_classes_dataset'
dataset = datasets.ImageFolder(root=path, transform=my_transform)

# Split dataset into train, test, and validation
train_size = int(0.7 * len(dataset))
test_size = int(0.15 * len(dataset))
valid_size = len(dataset) - train_size - test_size
train_set, test_set, val_set = random_split(dataset, [train_size, test_size, valid_size])

# Dataloaders
batch_size = 32
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size)
val_loader = DataLoader(val_set, batch_size=batch_size)

# Define CNN class
class CNN(nn.Module):
    def __init__(self, output_size=4):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # 3 input channels (RGB)
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128x128 -> 64x64

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x64 -> 32x32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32 -> 16x16
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),  # Flatten before FC
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)  # Output logits
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

model = CNN(output_size=4).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 5
for epoch in range(epochs):
    total_loss = 0
    model.train()

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# Validation Loop - Use to tune hyperparameters
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"correct: {correct}, total: {total}")
print(f"\nValidation Accuracy: {accuracy:.2f}%")


''' **Evaluate on test set after tuning hyperparameters on validation set**

# Test Loop - final evaluation on test data
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
print(f"correct: {correct}, total: {total}")
print(f"Test Accuracy: {test_accuracy:.2f}%")
'''
