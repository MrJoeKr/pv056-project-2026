import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class EmbeddingNet(nn.Module):
    """ResNet-50 backbone with embedding head for metric learning."""

    def __init__(self, embedding_dim: int = 512, pretrained: bool = True):
        super().__init__()
        # Load pretrained ResNet-50
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)

        # Remove the final FC layer
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # Output: (B, 2048, 1, 1)

        # Embedding head
        self.embedding_head = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1024, embedding_dim),
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
