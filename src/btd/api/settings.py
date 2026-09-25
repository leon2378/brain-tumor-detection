"""Service configuration from environment variables (prefix ``BTD_``) or a ``.env`` file."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="BTD_", env_file=".env", extra="ignore", protected_namespaces=("settings_",)
    )

    # Model
    model_path: Path = Field(Path("models/model.onnx"), description="ONNX model to serve")
    model_meta: Path | None = Field(None, description="model.json (defaults to the file next to the model)")
    providers: str = Field("auto", description="auto | cpu | cuda | tensorrt | comma-separated list")
    conf_threshold: float | None = Field(None, ge=0.0, le=1.0, description="override the tuned threshold")
    iou_threshold: float | None = Field(None, ge=0.0, le=1.0)
    max_det: int = Field(20, ge=1, le=300)
    intra_op_threads: int = Field(0, ge=0, description="0 = ONNX Runtime default")
    gpu_mem_limit_mb: int | None = Field(None, ge=256, description="cap the CUDA arena (e.g. on a 6 GB card)")
    verify_checksum: bool = True
    require_model: bool = Field(False, description="exit at startup if the model cannot be loaded")
    warmup_runs: int = Field(2, ge=0)

    # Request limits
    max_upload_mb: float = Field(10.0, gt=0)
    max_image_pixels: int = Field(25_000_000, gt=0, description="decompression-bomb guard")
    max_concurrency: int = Field(4, ge=1, description="concurrent inferences per worker")

    # Security / integration
    api_key: SecretStr | None = Field(None, description="if set, clients must send X-API-Key")
    cors_origins: list[str] = Field(
        default_factory=list, description='JSON list, e.g. ["http://localhost:5173"]'
    )
    enable_metrics: bool = True
    ui: bool = Field(True, description="serve the web page at / and its assets at /ui (off for API-only use)")
    sagemaker: bool = Field(False, description="add SageMaker's /ping and /invocations routes")

    # Logging
    log_level: str = "INFO"
    log_json: bool = True

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)
