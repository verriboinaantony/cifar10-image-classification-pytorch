from flask import Flask, request, render_template
import torch
import torchvision.transforms as transforms
from PIL import Image
import os

from model import CNN

# ------------------ CREATE APP ------------------
app = Flask(__name__)

# ------------------ LOAD MODEL ------------------
model = CNN()
model.load_state_dict(torch.load("cnn_model.pth", map_location="cpu"))
model.eval()

# ------------------ CLASS NAMES ------------------
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# ------------------ TRANSFORM ------------------
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

# ------------------ ROUTE ------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    
    if request.method == "POST":
        file = request.files["file"]
        
        if file:
            image = Image.open(file).convert("RGB")
            image = transform(image).unsqueeze(0)
            
            with torch.no_grad():
                outputs = model(image)
                _, predicted = torch.max(outputs, 1)
                prediction = classes[predicted.item()]
    
    return render_template("index.html", prediction=prediction)

# ------------------ RUN APP ------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)