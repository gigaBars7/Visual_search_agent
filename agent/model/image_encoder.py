import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoder(nn.Module):
    def __init__(
        self,
        image_size: int = 128,
        patch_size: int = 16,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        embedding_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        num_patches = (image_size // patch_size) ** 2
        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.position_embedding = nn.Parameter(
            torch.randn(1, num_patches, d_model) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.projection = nn.Linear(d_model, embedding_dim)

    def forward(self, images):
        tokens = self.patch_embedding(images)
        tokens = tokens.flatten(2).transpose(1, 2)
        tokens = tokens + self.position_embedding
        tokens = self.transformer(tokens)

        embedding = self.projection(tokens.mean(dim=1))
        return F.normalize(embedding, dim=-1)
