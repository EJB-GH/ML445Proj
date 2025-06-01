import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset
path = 'Alzheimer_MRI_4_classes_dataset'
dataset = datasets.ImageFolder(root=path, transform=transforms.ToTensor())

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

# ReLU function
def reLU(x):
    # Applies ReLU element-wise (replaces negative values with 0)
    return torch.maximum(x, torch.tensor(0.0, device=device))

# Softmax function
def softmax(x):
    # Applies softmax to each row (each sample in the batch)
    e_x = torch.exp(x - x.max(dim=1, keepdim=True).values)
    return e_x / e_x.sum(dim=1, keepdim=True)

# Cross-entropy loss (expects one-hot targets)
def cross_entropy(pred, target):
    eps = 1e-9  # avoid log(0)
    pred = torch.clamp(pred, eps, 1. - eps)
    return -torch.sum(target * torch.log(pred)) / pred.shape[0]

# Define the manual MLP
class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.hl_w = torch.empty(hidden_size, input_size, device=device).uniform_(-0.05, 0.05)
        self.hl_b = torch.zeros(1, hidden_size, device=device)
        self.ol_w = torch.empty(output_size, hidden_size, device=device).uniform_(-0.05, 0.05)
        self.ol_b = torch.zeros(1, output_size, device=device)

        # Momentum buffers
        self.hl_w_m = torch.zeros_like(self.hl_w)
        self.hl_b_m = torch.zeros_like(self.hl_b)
        self.ol_w_m = torch.zeros_like(self.ol_w)
        self.ol_b_m = torch.zeros_like(self.ol_b)

    def forward(self, x):
        # Flatten the images
        x = x.view(x.shape[0], -1).to(device)
        self.x = x  # Store input for backprop

        # Hidden layer (ReLU activation)
        self.hidden = reLU(x @ self.hl_w.T + self.hl_b)

        # Output layer (softmax for class probabilities)
        logits = self.hidden @ self.ol_w.T + self.ol_b
        self.output = softmax(logits)
        return self.output

    def backward(self, y_true, lr=0.08, momentum=0.05):
        # Convert labels to one-hot
        y_true = F.one_hot(y_true, num_classes=4).float().to(device)

        # Error from output
        error_output = self.output - y_true  # Shape: [batch, 4]

        # Gradients for output layer
        grad_ol_w = error_output.T @ self.hidden / self.x.shape[0]
        grad_ol_b = error_output.mean(dim=0, keepdim=True)

        # Error propagated to hidden layer
        d_hidden = error_output @ self.ol_w  # Shape: [batch, hidden]
        d_hidden[self.hidden <= 0] = 0  # ReLU derivative

        # Gradients for hidden layer
        grad_hl_w = d_hidden.T @ self.x / self.x.shape[0]
        grad_hl_b = d_hidden.mean(dim=0, keepdim=True)

        # Apply momentum and update weights
        self.ol_w_m = momentum * self.ol_w_m - lr * grad_ol_w
        self.ol_b_m = momentum * self.ol_b_m - lr * grad_ol_b
        self.hl_w_m = momentum * self.hl_w_m - lr * grad_hl_w
        self.hl_b_m = momentum * self.hl_b_m - lr * grad_hl_b

        self.ol_w += self.ol_w_m
        self.ol_b += self.ol_b_m
        self.hl_w += self.hl_w_m
        self.hl_b += self.hl_b_m

# Model initialization
input_size = 3 * 176 * 208 
hidden_size = 256
output_size = 4
model = MLP(input_size, hidden_size, output_size)

# Training loop
epochs = 5
for epoch in range(epochs):
    total_loss = 0

    for images, labels in train_loader:
        preds = model.forward(images)
        labels_onehot = F.one_hot(labels, num_classes=output_size).float().to(device)
        loss = cross_entropy(preds, labels_onehot)

        total_loss += loss.item()
        model.backward(labels)

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")