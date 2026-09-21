import torch
import torch.nn as nn
import torch.nn.functional as F


class TextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_length: int = 32,
        pad_token_id: int = 0,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        embedding_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=pad_token_id,
        )
        self.position_embedding = nn.Parameter(
            torch.randn(1, max_length, d_model) * 0.02
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

    def forward(self, input_ids, attention_mask):
        sequence_length = input_ids.shape[1]
        tokens = self.token_embedding(input_ids)
        tokens = tokens + self.position_embedding[:, :sequence_length]
        tokens = self.transformer(
            tokens,
            src_key_padding_mask=attention_mask == 0,
        )

        mask = attention_mask.unsqueeze(-1).float()
        embedding = (tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        embedding = self.projection(embedding)
        return F.normalize(embedding, dim=-1)
