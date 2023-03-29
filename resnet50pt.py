from __future__ import print_function
from __future__ import division
import torch.nn as nn
from torchvision import models

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(num_classes, use_pretrained=True, feature_extract=True):    
    model = models.resnet50(pretrained=use_pretrained)
    set_parameter_requires_grad(model, feature_extract)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model

def resnet50pt(device, num_classes=0):
    return initialize_model(num_classes).to(device)
