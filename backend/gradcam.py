import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms

from model import get_model

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Classes
classes = ['NORMAL', 'PNEUMONIA']

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models/model.pth")

# Load model
model = get_model(len(classes))

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model.to(device)

model.eval()


# Target layer (last convolution layer)
target_layer = None

for layer in reversed(model.features):
    if isinstance(layer, torch.nn.Conv2d):
        target_layer = layer
        break

if target_layer is None:
    raise ValueError("No Conv2d layer found in model.features")

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def generate_gradcam(image_path):

    gradients = []
    activations = []

    # Hooks
    def forward_hook(module, input, output):
        activations.append(output)

        if output.requires_grad:
            def grad_hook(grad):
                gradients.append(grad)

            output.register_hook(grad_hook)

    target_layer.register_forward_hook(forward_hook)

    # Load image
    image = Image.open(image_path).convert("RGB")

    input_tensor = transform(image).unsqueeze(0).to(device)

    # Enable gradients
    input_tensor.requires_grad_()

    # Forward pass
    output = model(input_tensor)

    pred_class = output.argmax(dim=1)

    # Backward pass
    model.zero_grad()

    output[0, pred_class.item()].backward()

    # Ensure hooks captured data
    if len(gradients) == 0:
        raise ValueError("Gradients were not captured. Check target layer.")

    if len(activations) == 0:
        raise ValueError("Activations were not captured. Check target layer.")

    # Get gradients and activations
    grads = gradients[0].detach().cpu().numpy()[0]

    acts = activations[0].detach().cpu().numpy()[0]

    # Compute weights
    weights = np.mean(grads, axis=(1, 2))

    cam = np.zeros(acts.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts[i]

    # ReLU
    cam = np.maximum(cam, 0)

    # Normalize
    if cam.max() != 0:
        cam = cam / cam.max()

    # Resize
    cam = cv2.resize(cam, (224, 224))

    # Original image
    original = cv2.imread(image_path)

    original = cv2.resize(original, (224, 224))

    # Heatmap
    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam),
        cv2.COLORMAP_JET
    )

    # Overlay
    overlay = cv2.addWeighted(
        original,
        0.6,
        heatmap,
        0.4,
        0
    )

    # Save result
    output_path = os.path.join(BASE_DIR, "gradcam_result.jpg")

    cv2.imwrite(output_path, overlay)

    # Show result
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

    plt.title(f"Prediction: {classes[pred_class.item()]}")

    plt.axis("off")

    plt.show()

    print(f"Grad-CAM saved at: {output_path}")





########## FIRST CODE

# import torch
# import cv2
# import numpy as np

# def generate_gradcam(model, image, target_layer):
#     gradients = []
#     activations = []

#     def backward_hook(module, grad_in, grad_out):
#         gradients.append(grad_out[0])

#     def forward_hook(module, input, output):
#         activations.append(output)

#     target_layer.register_forward_hook(forward_hook)
#     target_layer.register_backward_hook(backward_hook)

#     output = model(image)
#     pred_class = output.argmax()

#     output[:, pred_class].backward()

#     grads = gradients[0].cpu().data.numpy()[0]
#     acts = activations[0].cpu().data.numpy()[0]

#     weights = np.mean(grads, axis=(1, 2))
#     cam = np.zeros(acts.shape[1:], dtype=np.float32)

#     for i, w in enumerate(weights):
#         cam += w * acts[i]

#     cam = np.maximum(cam, 0)
#     cam = cv2.resize(cam, (224, 224))
#     cam = cam / cam.max()

#     return cam
