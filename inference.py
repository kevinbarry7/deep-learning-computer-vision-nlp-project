# Required imports


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvison.datasets as datasets

def net():
    
    # Function initializes model using a pretrained model


    num_classes = 133
    model = models.resnet50(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
    
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes),
    
    return model


def model_fn(model_dir):
    model = net()
    with open(os.path.join(model_dir, 'model_path'), 'rb') as f:
        model.load_state_dict(torch.load(f))

    return model


