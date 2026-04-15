import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


_BACKBONES = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT, 512),
    "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT, 512),
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT, 2048),
}


class EmbeddingNet(nn.Module):
    """ResNet backbone with embedding head for metric learning."""

    def __init__(self, embedding_dim: int = 512, pretrained: bool = True, backbone: str = "resnet50"):
        super().__init__()
        if backbone not in _BACKBONES:
            raise ValueError(f"Unknown backbone '{backbone}'. Supported: {list(_BACKBONES)}")
        constructor, default_weights, feature_dim = _BACKBONES[backbone]
        weights = default_weights if pretrained else None
        net = constructor(weights=weights)

        self.backbone = nn.Sequential(*list(net.children())[:-1])  # (B, feature_dim, 1, 1)

        self.embedding_head = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1024, embedding_dim),
        )

        self.embedding_dim = embedding_dim
        self.backbone_name = backbone
        self.feature_dim = feature_dim

    def forward(self, x):
        """Extract L2-normalized embeddings."""
        features = self.backbone(x)
        features = features.view(features.size(0), -1)  # Flatten
        embeddings = self.embedding_head(features)
        embeddings = F.normalize(embeddings, p=2, dim=1)  # L2 normalize
        return embeddings

    def get_backbone_params(self):
        """Get backbone parameters (for differential LR)."""
        return self.backbone.parameters()

    def get_head_params(self):
        """Get embedding head parameters."""
        return self.embedding_head.parameters()
