import torch
import torchvision.transforms as transforms
from PIL import Image

from model import CNN

# Class names
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Load model
model = CNN()
model.load_state_dict(torch.load("cnn_model.pth"))
model.eval()

# Transform (same as test)
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

# Load your image
image = Image.open("test_image.jpg")

# Apply transform
image = transform(image).unsqueeze(0)

# Predict
with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)

print("Predicted class:", classes[predicted.item()])