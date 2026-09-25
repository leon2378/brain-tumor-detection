"""Copy the web page's sample slices out of the prepared BRISC test split, with their expert outlines.

    python scripts/make_demo_samples.py [--data data/processed/brisc-yolo]

The slices are the typical cases shown in the README figure (scripts/make_readme_figure.py): the median-Dice correct
prediction of each tumour class and a correctly rejected healthy slice. Images are copied byte for byte; the expert
masks are turned into polygons with the same function the API uses for predicted masks. BRISC 2025 is CC BY 4.0, so
the page credits it wherever the samples appear.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from btd.inference.postprocess import mask_to_polygons
from btd.utils import imread_gray

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "src" / "btd" / "api" / "static" / "samples"
SAMPLES = [
    ("glioma", "Glioma", "brisc2025_test_00042_gl_ax_t1"),
    ("meningioma", "Meningioma", "brisc2025_test_00528_me_sa_t1"),
    ("pituitary", "Pituitary", "brisc2025_test_00769_pi_ax_t1"),
    ("no_tumor", "Healthy", "brisc2025_test_00561_no_ax_t1"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(REPO / "data" / "processed" / "brisc-yolo"))
    args = ap.parse_args()
    data = Path(args.data)

    OUT.mkdir(parents=True, exist_ok=True)
    entries = []
    for label, title, key in SAMPLES:
        image = data / "images" / "test" / f"{key}.jpg"
        shutil.copyfile(image, OUT / image.name)
        gray = imread_gray(image)
        mask_file = data / "masks" / "test" / f"{key}.png"
        polygons = mask_to_polygons(imread_gray(mask_file) > 127) if mask_file.is_file() else []
        entries.append(
            {
                "label": label,
                "title": title,
                "file": image.name,
                "width": int(gray.shape[1]),
                "height": int(gray.shape[0]),
                "expert_polygons": polygons,
            }
        )
        print(f"  {title:<11} {image.name}  ({len(polygons)} expert polygon(s))")

    manifest = {
        "source": "BRISC 2025 test split, CC BY 4.0, https://doi.org/10.5281/zenodo.17524350",
        "samples": entries,
    }
    (OUT / "samples.json").write_text(json.dumps(manifest, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {OUT / 'samples.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
