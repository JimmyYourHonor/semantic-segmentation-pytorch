import torch
import torch.nn as nn
import numpy as np

def min_max_normalize(image):
    min_val = np.min(image)
    max_val = np.max(image)
    normalized_image = (image - min_val) / (max_val - min_val)
    return normalized_image

def get_activation_hook(features, name):
    def forward_pre_hook(module, inputs):
        features[name] = []
        for input in inputs:
            input.requires_grad_()
            features[name].append(input)
    return forward_pre_hook