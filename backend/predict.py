import os
import torch
from torchvision import transforms
from PIL import Image
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

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict(image_path):

    image = Image.open(image_path).convert("RGB")

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(image)

        probabilities = torch.softmax(output, dim=1)

        confidence, pred = torch.max(probabilities, 1)

    predicted_class = classes[pred.item()]

    confidence_score = confidence.item() * 100

    return predicted_class, confidence_score



### first code

# import torch
# from torchvision import transforms
# from PIL import Image
# from model import get_model

# classes = ['normal', 'pneumonia']  # example

# def predict(image_path):
#     model = get_model(len(classes))
#     model.load_state_dict(torch.load("models/model.pth"))
#     model.eval()

#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor()
#     ])

#     image = Image.open(image_path).convert("RGB")
#     image = transform(image).unsqueeze(0)

#     output = model(image)
#     _, pred = torch.max(output, 1)

#     return classes[pred.item()]