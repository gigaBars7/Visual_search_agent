import torch.nn as nn

from .image_encoder import ImageEncoder
from .text_encoder import TextEncoder


class MiniCLIP(nn.Module):
    def __init__(self, image_encoder: ImageEncoder, text_encoder: TextEncoder) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

    def encode_image(self, images):
        return self.image_encoder(images)

    def encode_text(self, input_ids, attention_mask):
        return self.text_encoder(input_ids, attention_mask)
