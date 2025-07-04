import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),  
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

    def train_model(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0008)
        #optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        epochs = 50
        total_valid = []
        total_test = []

        # For confusion matrix
        test_preds = []
        test_trues = []

        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            self.train()

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                # Forward pass
                outputs = self(images)
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
        
            # Evaluate models
            val_acc,_,_ = self.eval_model(val_loader, "Validation")
            test_acc, test_preds, test_trues = self.eval_model(test_loader, "Test")

            total_valid.append(val_acc)
            total_test.append(test_acc)

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f} Validation Accuracy: {val_acc:.2f} Test Accuracy: {test_acc:.2f}")

        # Confusion matrix and plot
        self.computeConfusionMatrix(test_preds, test_trues) 
        self.computePlot(total_valid, total_test)

    def eval_model(self, loader, type_loader):
        self.eval()
        correct = 0
        total = 0
        pred_list = []
        true_list = []

        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                pred_list.extend(predicted.cpu().tolist())
                true_list.extend(labels.cpu().tolist())

        accuracy = 100 * correct / total
        print(f"{type_loader} - correct: {correct}, total: {total}")
        return accuracy, pred_list, true_list
    
    def computeConfusionMatrix(self, true_values, pred_values):
        cm = confusion_matrix(true_values, pred_values)
        display_matrix = ConfusionMatrixDisplay(confusion_matrix=cm)
        display_matrix.plot(cmap='Blues')
        plt.title("Confusion Matrix")
        plt.show()
    
    def computePlot(self, total_valid, total_test):
        plt.plot(total_valid, label = 'Validation')
        plt.plot(total_test, label = 'Test')
        plt.title("Validation vs Test Accuracy")
        plt.legend()
        plt.ylabel('Accuracy (%)')
        plt.xlabel('Epoch')
        plt.ylim(40,100)
        plt.show() 

model = CNN(output_size=4).to(device)
model.train_model()