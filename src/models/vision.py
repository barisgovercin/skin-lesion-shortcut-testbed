import torch
import torch.nn as nn
from torchvision import models


class VisionClassifier(nn.Module):
    def __init__(self, backbone="convnext_large", pretrained=True, num_classes=1):
        super().__init__()

        if backbone == "convnext_large":
            weights = models.ConvNeXt_Large_Weights.IMAGENET1K_V1 if pretrained else None
            self.encoder = models.convnext_large(weights=weights)
            in_features = self.encoder.classifier[2].in_features
            self.encoder.classifier[2] = nn.Identity()
        elif backbone == "convnext_base":
            weights = models.ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
            self.encoder = models.convnext_base(weights=weights)
            in_features = self.encoder.classifier[2].in_features
            self.encoder.classifier[2] = nn.Identity()
        elif backbone == "efficientnet_b3":
            weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
            self.encoder = models.efficientnet_b3(weights=weights)
            in_features = self.encoder.classifier[1].in_features
            self.encoder.classifier = nn.Identity()
        elif backbone == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            self.encoder = models.resnet50(weights=weights)
            in_features = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.classifier = nn.Linear(in_features, num_classes)
        self.feature_dim = in_features

    def forward(self, x):
        features = self.encoder(x)
        return self.classifier(features).squeeze(-1)

    def extract_features(self, x):
        return self.encoder(x)
