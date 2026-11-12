"""Backbone creation and preprocessing helpers built on timm."""
import timm
import torch


def create_backbone(name: str, pretrained: bool = True, **model_kwargs) -> torch.nn.Module:
    """Create a timm model that outputs one pooled feature vector per image.

    `num_classes=0` removes the classification head, so the model returns the
    globally pooled features (global average pooling for ConvNeXt,
    the class token for ViT models).
    """
    return timm.create_model(name, pretrained=pretrained, num_classes=0, **model_kwargs)


def build_transform(model: torch.nn.Module, input_size: int | None = None,
                    crop_mode: str = "squash"):
    """Build the evaluation preprocessing pipeline for a model.

    Mean, std and interpolation come from the model's pretrained config.
    crop_mode="squash" resizes the whole image to a square; "center" resizes the shorter side
    and center-crops.
    """
    args = {"input_size": (3, input_size, input_size)} if input_size else {}
    config = timm.data.resolve_data_config(args, model=model)
    config["crop_pct"] = 1.0
    config["crop_mode"] = crop_mode
    return timm.data.create_transform(**config, is_training=False)
