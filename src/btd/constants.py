"""Project-wide constants: label spaces, BRISC 2025 filename codes and dataset provenance."""

from __future__ import annotations

from typing import Final

# Detector classes, in YOLO class-id order. "No tumour" is not a detector class: healthy slices are trained as
# background images (empty label files), and the image-level "no_tumor" label is derived from the absence of
# confident detections (see btd.inference.types.ImageDecision).
TUMOR_CLASSES: Final[tuple[str, ...]] = ("glioma", "meningioma", "pituitary")
NO_TUMOR: Final[str] = "no_tumor"
IMAGE_LABELS: Final[tuple[str, ...]] = (*TUMOR_CLASSES, NO_TUMOR)

# BRISC 2025 filename convention: brisc2025_<split>_<index>_<tumor>_<plane>_<sequence>.<ext>
TUMOR_CODES: Final[dict[str, str]] = {
    "gl": "glioma",
    "me": "meningioma",
    "pi": "pituitary",
    "no": NO_TUMOR,
}
PLANE_CODES: Final[dict[str, str]] = {"ax": "axial", "co": "coronal", "sa": "sagittal"}

# Aliases seen in folder names of common Kaggle brain-tumour datasets -> canonical labels.
LABEL_ALIASES: Final[dict[str, str]] = {
    "glioma": "glioma",
    "glioma_tumor": "glioma",
    "meningioma": "meningioma",
    "meningioma_tumor": "meningioma",
    "pituitary": "pituitary",
    "pituitary_tumor": "pituitary",
    "notumor": NO_TUMOR,
    "no_tumor": NO_TUMOR,
    "no tumor": NO_TUMOR,
    "no": NO_TUMOR,
    "healthy": NO_TUMOR,
    "normal": NO_TUMOR,
}

IMAGE_EXTENSIONS: Final[frozenset[str]] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})

# Dataset provenance (CC BY 4.0 — attribution required, see README).
BRISC_ZENODO_RECORD: Final[str] = "17524350"
BRISC_ZIP_NAME: Final[str] = "brisc2025.zip"
# A SHA-256 for every file; Zenodo publishes it next to the zip, not inside it.
BRISC_MANIFEST_NAME: Final[str] = "manifest.csv"
BRISC_DOI: Final[str] = "10.5281/zenodo.17524350"
BRISC_KAGGLE: Final[str] = "briscdataset/brisc2025"
BRISC_CITATION: Final[str] = (
    "Fateh, A. et al. BRISC: Annotated Dataset for Brain Tumor Segmentation and Classification. "
    "Scientific Data (2026). https://doi.org/10.1038/s41597-026-06753-y"
)

DISCLAIMER: Final[str] = (
    "Research/educational prototype. Not a medical device and not validated for clinical diagnosis."
)


def canonical_label(name: str) -> str | None:
    """Map a folder name such as ``notumor`` or ``glioma_tumor`` to a canonical image label."""
    return LABEL_ALIASES.get(name.strip().lower().replace("-", "_"))
