"""Brain tumour detection & segmentation on MRI: data pipeline, training, quantised export and serving."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("brain-tumor-detection")
except PackageNotFoundError:  # pragma: no cover - running from a source checkout without install
    __version__ = "0.0.0+local"

__all__ = ["__version__"]
