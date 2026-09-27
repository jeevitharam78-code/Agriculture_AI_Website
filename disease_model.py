import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# -----------------------------
# File paths
# -----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "plant_disease_model.pth"
)

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR,
    "models",
    "class_names.txt"
)


# -----------------------------
# Load class names
# -----------------------------

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    class_names = [
        line.strip()
        for line in f
        if line.strip()
    ]


# -----------------------------
# Use CPU for Plant Disease
# -----------------------------
# Gemma uses the RTX 3050 GPU.
# Plant Disease runs on CPU to avoid GPU VRAM conflict.

device = torch.device("cpu")

print("Plant Disease Device:", device)


# -----------------------------
# Create EfficientNet-B0
# -----------------------------

model = models.efficientnet_b0(weights=None)

num_classes = len(class_names)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    num_classes
)


# -----------------------------
# Load trained model
# -----------------------------

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    checkpoint = checkpoint["state_dict"]

model.load_state_dict(checkpoint)

model = model.to(device)

model.eval()


# -----------------------------
# Image preprocessing
# -----------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -----------------------------
# Prediction function
# -----------------------------

def predict_disease(image_path):

    image = Image.open(image_path).convert("RGB")

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            1
        )

    predicted_class = class_names[
        predicted.item()
    ]

    confidence_value = confidence.item() * 100

    return predicted_class, confidence_value