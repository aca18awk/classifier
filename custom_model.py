import torch.nn as nn
import timm

class GlaucomaClassifier(nn.Module):
    def __init__(self, model_name="efficientnet_b0", num_classes=3):
        super(GlaucomaClassifier, self).__init__()
        self.base_model = timm.create_model(
            model_name, 
            pretrained=True, 
            num_classes=num_classes
        )

    def forward(self, x):
        return self.base_model(x)

