from flask import Flask, render_template, request
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# ===== Initialize Flask app =====
app = Flask(__name__)

# ===== Ensure static folder exists =====
os.makedirs("static", exist_ok=True)

# ===== Load model (ResNet34 same as training) =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet34(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load("cancer_model_v2.pth", map_location=device))
model = model.to(device)
model.eval()

# ===== Define image transformations =====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ===== Home page =====
@app.route('/')
def home():
    return render_template('index.html')

# ===== Prediction route =====
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', message="❌ No file uploaded!")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', message="❌ No file selected!")

    # Save uploaded file to static folder
    filename = file.filename
    file_path = os.path.join("static", filename)
    os.makedirs("static", exist_ok=True)
    file.save(file_path)

    # Open and preprocess the image
    image = Image.open(file_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        label = "🧬 Cancer" if preds.item() == 0 else "✅ Non-Cancer"

    # Show result on webpage
    return render_template('index.html', 
                           prediction=label, 
                           image_path=file_path)

# ===== Run the app =====
if __name__ == '__main__':
    app.run(debug=True)
