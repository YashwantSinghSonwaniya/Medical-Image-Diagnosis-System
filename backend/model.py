import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import AlexNet_Weights

def get_model(num_classes):

    weights = None

    model = models.alexnet(weights=None)

    # Freeze feature extractor
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace classifier
    model.classifier[6] = nn.Linear(4096, num_classes)

    return model





### first code

# import torch
# import torch.nn as nn
# from torchvision import models

# def get_model(num_classes):
#     model = models.alexnet(pretrained=True)

#     # Freeze layers
#     for param in model.features.parameters():
#         param.requires_grad = False

#     # Modify classifier
#     model.classifier[6] = nn.Linear(4096, num_classes)

#     return model