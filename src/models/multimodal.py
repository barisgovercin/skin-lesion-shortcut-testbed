"""Multimodal model: vision + optional metadata + optional text, late fusion."""

import torch
import torch.nn as nn
from src.models.vision import VisionClassifier


class MetadataEncoder(nn.Module):
    """Simple MLP encoder for tabular metadata features."""

    def __init__(self, input_dim, hidden_dim=128, output_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )
        self.feature_dim = output_dim

    def forward(self, x):
        return self.encoder(x)


class MultimodalClassifier(nn.Module):
    """Late fusion of vision + optional metadata + optional text features.

    Architecture follows Watson et al. (2026): each modality is encoded
    independently, the feature vectors are concatenated, and a single linear
    head produces the logit. Branches are toggled so one class covers every
    combination (vision+metadata, vision+text, vision+metadata+text).

    For offline testing, a pre-built `text_encoder` module can be injected so
    the text branch is exercised without downloading BioClinicalBERT.
    """

    def __init__(
        self,
        backbone="convnext_base",
        pretrained=True,
        use_metadata=True,
        metadata_dim=12,
        metadata_hidden=128,
        metadata_output=64,
        use_text=False,
        text_model_name="emilyalsentzer/Bio_ClinicalBERT",
        text_pretrained=True,
        text_freeze=False,
        text_encoder=None,
        num_classes=1,
    ):
        super().__init__()
        self.use_metadata = use_metadata
        self.use_text = use_text

        # Vision encoder (always present); drop its vision-only head.
        self.vision = VisionClassifier(
            backbone=backbone, pretrained=pretrained, num_classes=1
        )
        fused_dim = self.vision.feature_dim
        self.vision.classifier = nn.Identity()

        if use_metadata:
            self.metadata_encoder = MetadataEncoder(
                input_dim=metadata_dim,
                hidden_dim=metadata_hidden,
                output_dim=metadata_output,
            )
            fused_dim += self.metadata_encoder.feature_dim

        if use_text:
            if text_encoder is None:
                from src.models.text import TextEncoder

                text_encoder = TextEncoder(
                    model_name=text_model_name,
                    pretrained=text_pretrained,
                    freeze=text_freeze,
                )
            self.text_encoder = text_encoder
            fused_dim += self.text_encoder.feature_dim

        self.classifier = nn.Linear(fused_dim, num_classes)

    def extract_features(self, image, metadata=None, text=None):
        feats = [self.vision(image)]
        if self.use_metadata:
            feats.append(self.metadata_encoder(metadata))
        if self.use_text:
            input_ids, attention_mask = text
            feats.append(self.text_encoder(input_ids, attention_mask))
        return torch.cat(feats, dim=1)

    def forward(self, image, metadata=None, text=None):
        fused = self.extract_features(image, metadata=metadata, text=text)
        return self.classifier(fused).squeeze(-1)
