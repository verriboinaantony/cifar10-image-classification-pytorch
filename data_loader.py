import torch
import torchvision
import torchvision.transforms as transforms

# ------------------ TRAIN TRANSFORMS ------------------
# These are applied only to training data
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),          # Random flip (left-right)
    transforms.RandomCrop(32, padding=4),       # Random crop with padding
    
    transforms.ToTensor(),                      # Convert image to tensor
    transforms.Normalize((0.5, 0.5, 0.5), 
                         (0.5, 0.5, 0.5))       # Normalize values
])

# ------------------ TEST TRANSFORMS ------------------
# No augmentation for test data
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), 
                         (0.5, 0.5, 0.5))
])

# ------------------ LOAD DATASETS ------------------
train_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=train_transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=test_transform
)

# ------------------ DATALOADERS ------------------
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

# ------------------ CLASS NAMES ------------------
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

print("Dataset with augmentation loaded successfully!")