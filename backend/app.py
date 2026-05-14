import gdown
import os
import cv2
import torch
import tempfile
import numpy as np
import streamlit as st

from PIL import Image
from torchvision import transforms

from model import get_model

# ---------------------------
# CONFIG
# ---------------------------

st.set_page_config(
    page_title="Medical Image Diagnosis System",
    layout="centered"
)

st.title("🩺 Medical Image Diagnosis System")

st.write("Upload a chest X-ray image for pneumonia detection.")

# ---------------------------
# DEVICE
# ---------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ---------------------------
# CLASSES
# ---------------------------

classes = ['NORMAL', 'PNEUMONIA']

# ---------------------------
# PATHS
# ---------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models/model.pth"
)

if not os.path.exists(MODEL_PATH):

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    file_id = "1SnnIgrOOxrtBMrK3lR0zhm_B4Ew1rLVg"

    url = f"https://drive.google.com/uc?id={file_id}"

    gdown.download(url, MODEL_PATH, quiet=False)

# ---------------------------
# LOAD MODEL
# ---------------------------

model = get_model(len(classes))

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model.to(device)

model.eval()

# ---------------------------
# TARGET LAYER
# ---------------------------

target_layer = None

for layer in reversed(model.features):
    if isinstance(layer, torch.nn.Conv2d):
        target_layer = layer
        break

if target_layer is None:
    raise ValueError("No Conv2d layer found in model.features")

# ---------------------------
# TRANSFORM
# ---------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ---------------------------
# GRADCAM FUNCTION
# ---------------------------

def generate_gradcam(image_path):

    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

        if output.requires_grad:
            def grad_hook(grad):
                gradients.append(grad)

            output.register_hook(grad_hook)

    target_layer.register_forward_hook(forward_hook)

    image = Image.open(image_path).convert("RGB")

    input_tensor = transform(image).unsqueeze(0).to(device)

    input_tensor.requires_grad_()

    output = model(input_tensor)

    probabilities = torch.softmax(output, dim=1)

    confidence, pred = torch.max(probabilities, 1)

    model.zero_grad()

    output[0, pred.item()].backward()

    if len(gradients) == 0:
        raise ValueError("Gradients were not captured. Check target layer.")

    if len(activations) == 0:
        raise ValueError("Activations were not captured. Check target layer.")

    grads = gradients[0].detach().cpu().numpy()[0]

    acts = activations[0].detach().cpu().numpy()[0]

    weights = np.mean(grads, axis=(1, 2))

    cam = np.zeros(acts.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)

    if cam.max() != 0:
        cam = cam / cam.max()

    cam = cv2.resize(cam, (224, 224))

    original = cv2.imread(image_path)

    original = cv2.resize(original, (224, 224))

    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam),
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(
        original,
        0.6,
        heatmap,
        0.4,
        0
    )

    return (
        overlay,
        classes[pred.item()],
        confidence.item() * 100
    )

# ---------------------------
# FILE UPLOAD
# ---------------------------

uploaded_file = st.file_uploader(
    "Upload X-ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    st.image(uploaded_file, caption="Uploaded Image")

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg"
    ) as tmp_file:

        tmp_file.write(uploaded_file.read())

        temp_path = tmp_file.name

    overlay, prediction, confidence = generate_gradcam(temp_path)

    st.subheader(f"Prediction: {prediction}")

    st.subheader(f"Confidence: {confidence:.2f}%")

    st.image(
        cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
        caption="Grad-CAM Visualization"
    )