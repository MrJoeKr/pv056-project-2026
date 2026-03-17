import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class EmbeddingNet(nn.Module):
    """ResNet backbone with embedding head for metric learning."""

    def __init__(self, embedding_dim: int = 512, pretrained: bool = True, backbone_name: str = "resnet50"):
        super().__init__()
        # Load pretrained backbone
        if backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            net = models.resnet50(weights=weights)
            feature_dim = 2048
        elif backbone_name == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            net = models.resnet34(weights=weights)
            feature_dim = 512
        elif backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            net = models.resnet18(weights=weights)
            feature_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # Remove the final FC layer
        self.backbone = nn.Sequential(*list(net.children())[:-1])  # Output: (B, feature_dim, 1, 1)

        # Embedding head (adaptive to feature_dim)
        hidden_dim = max(512, feature_dim // 2)
        self.embedding_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, embedding_dim),
        )

        self.embedding_dim = embedding_dim

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
