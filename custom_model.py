import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm
from typing import Literal
from enum import Enum

import matplotlib.pyplot as plt # For data viz
import pandas as pd
import numpy as np
import sys
import os

class GlaucomaClassifier(nn.Module):
    def __init__(self, model_name="efficientnet_b0", num_classes=3):
        super(GlaucomaClassifier, self).__init__()
        # timm does the magic for you:
        self.base_model = timm.create_model(
            model_name, 
            pretrained=True, 
            num_classes=num_classes # <--- This handles the head automatically
        )

    def forward(self, x):
        return self.base_model(x)
    # def __init__(self, model_name = "efficientnet_b0", num_classes = 3):
    #     super(GlaucomaClassifier, self).__init__()

    #     self.base_model = timm.create_model(model_name, pretrained=True)
    #     print(self.base_model.children())
    #     # remove the last output layer with default number of classes
    #     self.features = nn.Sequential(*list(self.base_model.children())[:-1])

    #     out_size = 1280

    #     # define the new last layer that will output the correct classes
    #     self.classifier = nn.Linear(out_size, num_classes)

    # def forward(self, x):
    #     x = self.features(x)
    #     output = self.classifier(x)
    #     return output
    

# GlaucomaClassifier()

