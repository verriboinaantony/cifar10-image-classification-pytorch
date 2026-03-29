# import torch
# import torch.nn as nn
# import torch.optim as optim

# from data_loader import train_loader, test_loader
# from model import MLP, CNN

# # Step 1: Device (CPU/GPU)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Step 2: Load model
# model = CNN().to(device)

# # Step 3: Loss function
# criterion = nn.CrossEntropyLoss()

# # Step 4: Optimizer
# optimizer = optim.Adam(model.parameters(), lr=0.0005)

# # Step 5: Training loop
# epochs = 20

# for epoch in range(epochs):
#     running_loss = 0.0
    
#     for images, labels in train_loader:
        
#         # Move data to device
#         images, labels = images.to(device), labels.to(device)
        
#         # Zero gradients
#         optimizer.zero_grad()
        
#         # Forward pass
#         outputs = model(images)
        
#         # Compute loss
#         loss = criterion(outputs, labels)
        
#         # Backward pass
#         loss.backward()
        
#         # Update weights
#         optimizer.step()
        
#         running_loss += loss.item()
    
#     print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss:.4f}")

# print("Training finished!")

# correct = 0
# total = 0

# with torch.no_grad():
#     for images, labels in test_loader:
#         images, labels = images.to(device), labels.to(device)
        
#         outputs = model(images)
#         _, predicted = torch.max(outputs, 1)
        
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# accuracy = 100 * correct / total
# print(f"Test Accuracy: {accuracy:.2f}%")




import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from data_loader import train_loader, test_loader
from model import MLP, CNN

# ------------------ DEVICE ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ SELECT MODEL ------------------
# Change this to MLP() if you want to test baseline
model = CNN().to(device)

# ------------------ LOSS & OPTIMIZER ------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# ------------------ TRAINING SETTINGS ------------------
epochs = 30
train_losses = []

# ------------------ TRAINING LOOP ------------------
for epoch in range(epochs):
    running_loss = 0.0
    
    for images, labels in train_loader:
        
        # Move data to device
        images, labels = images.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Compute loss
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        running_loss += loss.item()
    
    # Average loss per epoch
    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")

print("Training finished!")

# ------------------ EVALUATION ------------------
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

torch.save(model.state_dict(), "cnn_model.pth")
print("Model saved successfully!")

# ------------------ PLOT LOSS CURVE ------------------
plt.plot(train_losses, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.show()

