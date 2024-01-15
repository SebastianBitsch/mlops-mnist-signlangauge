import torch.nn as nn
import torch
import timm

# model = timm.create_model(model_name, num_classes=num_classes, pretrained=pretrained)

def get_timm():
    model = timm.create_model('resnet34', num_classes=25)
    return model


