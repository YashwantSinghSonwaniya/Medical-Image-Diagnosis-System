import os
from gradcam import generate_gradcam

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

image_path = os.path.join(
    BASE_DIR,
    "..", # currently in medical-image-diagnosis-system/backend, so go one level up
    "my_Images",
    "img1.jpg"
)

generate_gradcam(image_path)




####### FIRST CODE

# from gradcam import generate_gradcam

# image_path = "../chest_xray/test/PNEUMONIA/person1_virus_6.jpeg"

# generate_gradcam(image_path)