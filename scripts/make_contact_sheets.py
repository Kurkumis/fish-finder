"""Create one contact sheet of random photos per class for visual inspection."""
import pandas as pd
from PIL import Image, ImageOps

from fishfinder.utils.paths import INAT_RAW_DIR, INTERIM_DIR, PROCESSED_DIR

THUMB = 220  # thumbnail size in pixels
COLS, ROWS = 6, 4  # 24 photos per sheet


def main() -> None:
    df = pd.read_csv(PROCESSED_DIR / "dataset.csv")
    out_dir = INTERIM_DIR / "contact_sheets"
    out_dir.mkdir(parents=True, exist_ok=True)

    for class_id, group in df.groupby("class_id", sort=False):
        sample = group.sample(n=min(COLS * ROWS, len(group)), random_state=0)
        sheet = Image.new("RGB", (COLS * THUMB, ROWS * THUMB), "white")
        for k, path in enumerate(sample["local_path"]):
            with Image.open(INAT_RAW_DIR / path) as img:
                img = ImageOps.exif_transpose(img).convert("RGB")  # apply camera rotation
                img.thumbnail((THUMB, THUMB))
                x = (k % COLS) * THUMB + (THUMB - img.width) // 2
                y = (k // COLS) * THUMB + (THUMB - img.height) // 2
                sheet.paste(img, (x, y))
        sheet.save(out_dir / f"{class_id}.jpg", quality=85)
        print(f"saved {class_id}")


if __name__ == "__main__":
    main()