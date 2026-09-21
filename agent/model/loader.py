from pathlib import Path

import torch
from torchvision import transforms
from transformers import AutoTokenizer

from .image_encoder import ImageEncoder
from .mini_clip import MiniCLIP
from .text_encoder import TextEncoder


def _models_dir():
    container_models_dir = Path("/app/models")
    if container_models_dir.is_dir():
        return container_models_dir
    return Path(__file__).resolve().parents[2] / "models"


class LoadedMiniCLIP:
    def __init__(
        self,
        model,
        tokenizer,
        image_transform,
        device,
        max_text_length,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.device = device
        self.max_text_length = max_text_length

    def encode_image(self, image):
        image_tensor = self.image_transform(image.convert("RGB"))
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        with torch.inference_mode():
            return self.model.encode_image(image_tensor).squeeze(0).cpu()

    def encode_text(self, text):
        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = tokens["attention_mask"].to(self.device)
        with torch.inference_mode():
            return self.model.encode_text(input_ids, attention_mask).squeeze(0).cpu()


def load_model(
    model_path=None,
    tokenizer_path=None,
    device=None,
):
    models_dir = _models_dir()
    model_path = model_path or models_dir / "model" / "model_30k.pt"
    tokenizer_path = tokenizer_path or models_dir / "tokenizer"
    selected_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    artifact = torch.load(model_path, map_location=selected_device, weights_only=True)
    model_config = artifact["model_config"]
    text_config = artifact["text_config"]

    image_encoder = ImageEncoder(**model_config)
    text_encoder = TextEncoder(
        vocab_size=text_config["vocab_size"],
        max_length=text_config["max_length"],
        pad_token_id=text_config["pad_token_id"],
        d_model=model_config["d_model"],
        num_heads=model_config["num_heads"],
        num_layers=model_config["num_layers"],
        embedding_dim=model_config["embedding_dim"],
        dropout=model_config["dropout"],
    )
    image_encoder.load_state_dict(artifact["image_encoder_state_dict"])
    text_encoder.load_state_dict(artifact["text_encoder_state_dict"])

    model = MiniCLIP(image_encoder, text_encoder).to(selected_device)
    model.eval()

    normalization = artifact["image_normalization"]
    image_transform = transforms.Compose(
        [
            transforms.Resize(144),
            transforms.CenterCrop(model_config["image_size"]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=normalization["mean"],
                std=normalization["std"],
            ),
        ]
    )

    return LoadedMiniCLIP(
        model=model,
        tokenizer=AutoTokenizer.from_pretrained(tokenizer_path),
        image_transform=image_transform,
        device=selected_device,
        max_text_length=text_config["max_length"],
    )
