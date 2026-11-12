"""PyTorch dataset over the photo table."""
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps
from torch.utils.data import Dataset


class FishDataset(Dataset):
    """Photos listed in a dataset table, loaded lazily from disk."""

    def __init__(self, frame: pd.DataFrame, root: Path, transform=None):
        self.paths = [root / p for p in frame["local_path"]]
        self.labels = frame["class_index"].to_numpy()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        with Image.open(self.paths[idx]) as img:
            # Apply the camera rotation stored in EXIF so the fish is upright.
            img = ImageOps.exif_transpose(img).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[idx])
    