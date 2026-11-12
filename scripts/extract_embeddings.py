"""Extract embeddings of all dataset photos with frozen pretrained backbones."""
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from fishfinder.data_pipeline.dataset import FishDataset
from fishfinder.models.backbone import build_transform, create_backbone
from fishfinder.utils.config import load_backbones
from fishfinder.utils.paths import ARTIFACTS_DIR, INAT_RAW_DIR, PROCESSED_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop-mode", default="squash", choices=["squash", "center"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--force", action="store_true", help="recompute existing embeddings")
    args = parser.parse_args()

    df = pd.read_csv(PROCESSED_DIR / "dataset.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ARTIFACTS_DIR / "embeddings" / args.crop_mode
    out_dir.mkdir(parents=True, exist_ok=True)

    for entry in load_backbones():
        name = entry["name"]
        out_path = out_dir / f"{name}.npy"
        if out_path.exists() and not args.force:
            print(f"skip {name}: already computed")
            continue

        model = create_backbone(name, **entry.get("model_kwargs", {})).to(device).eval()
        transform = build_transform(model, entry.get("input_size"), args.crop_mode)
        loader = DataLoader(
            FishDataset(df, INAT_RAW_DIR, transform),
            batch_size=args.batch_size,
            num_workers=args.workers,
            shuffle=False,  # keep the row order: embeddings must align with dataset.csv
        )

        chunks = []
        with torch.inference_mode(), torch.autocast(device.type, dtype=torch.float16):
            for images, _ in tqdm(loader, desc=name):
                features = model(images.to(device))
                chunks.append(features.float().cpu())

        embeddings = torch.cat(chunks).numpy().astype(np.float32)
        np.save(out_path, embeddings)
        print(f"saved {name}: {embeddings.shape} -> {out_path}")

        del model  # free GPU memory before the next backbone
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
