# Quantisation and the 6 GB VRAM budget

## Which artefact to deploy where

| Artefact | Built by | Best on | Notes |
|---|---|---|---|
| `model_fp32.onnx` | Ultralytics ONNX export (opset 18, onnxslim) | reference, **CPU image** | Used to tune the threshold and to measure the accuracy of every other precision. The published CPU image ships this one: INT8 is faster on CPU but costs 3.4 points of image-level accuracy |
| `model_fp16.onnx` | `onnxruntime.transformers.float16` (FP32 I/O kept) | **CUDA GPU** | About 2× smaller; tensor-core friendly. Needs only a few hundred MB of VRAM |
| `model_int8.onnx` | ONNX Runtime static QDQ quantisation (`btd.export.quantize`) | **x86 CPU** | About 3.2× smaller. U8S8, per-channel weights, calibrated on 256 train images |
| `model_fp16.engine` / `model_int8.engine` (optional) | Ultralytics → TensorRT | the exact GPU it was built on | Fastest on NVIDIA, but not portable across GPUs or driver versions |

## Why it fits in 6 GB

- **Training** uses AMP (FP16 mixed precision). YOLO26s-seg at 640 px with batch 8 peaked at 3.13 GB on a 6 GB
  RTX 3060 Laptop GPU (`peak_gpu_memory_reserved_gb` in `run_summary.json`). If you hit OOM, use `batch=4`, or
  `model=yolo26n-seg.pt`, or `imgsz=512`.
- **Inference** in FP16 needs well under 1 GB. The service caps the ONNX Runtime CUDA arena with
  `BTD_GPU_MEM_LIMIT_MB` (2048 in the GPU image), so the GPU stays shareable.

## How the INT8 model is built

1. **Calibration data** is 256 images from the *train* split (never val or test), stratified across the four
   labels including no-tumour slices. They go through **the same letterbox pre-processing as the server**, so the
   activation ranges match what production inputs look like.
2. `quant_pre_process` runs shape inference and graph optimisation first.
3. **What gets quantised:** the whole backbone, neck and head. Everything *downstream* of the head's final
   convolutions stays FP32: box decoding, sigmoid scores, TopK/Gather and mask-coefficient plumbing. A single INT8
   scale can't represent box coordinates (0–640 px) and probabilities (0–1) at once.
4. **U8S8** (uint8 activations, int8 per-channel symmetric weights), QDQ format, MinMax calibration. You can switch
   to `percentile` or `entropy` in `configs/export.yaml`, and `keep_head_fp32: true` keeps the final head
   convolutions in FP32.
5. `target: tensorrt` produces symmetric int8 activations, which TensorRT's explicit-quantisation path needs.

## A measured pitfall: the default INT8 export can be *slower* than FP32

This was measured while building the pipeline on **YOLO26n-seg** (COCO weights) at 640×640, ONNX Runtime 1.30, a
2-vCPU Intel Xeon with AVX-512 VNNI. Times are for the model only (median of 10 runs):

| Variant | Conv nodes fused to `QLinearConv` | Latency |
|---|---|---|
| FP32 | — | 68 ms |
| INT8, S8S8, only Conv/MatMul quantised (the ORT/Ultralytics defaults) | 15 / 117 | 174–189 ms |
| INT8, **U8S8**, whole backbone quantised (this repo) | **117 / 117** | **49 ms** |

On x86, ONNX Runtime's QDQ fusion turns `DequantizeLinear → Conv → QuantizeLinear` into a real `QLinearConv` only
for U8S8. With int8 activations (S8S8), and with activations left in float between convolutions, most convolutions
fall back to FP32 compute *plus* quantise/dequantise overhead. On the same sample images the U8S8 model kept the
mask IoU against FP32 at 0.97–0.98.

Measure on your own hardware with:

```powershell
btd benchmark --providers cpu cuda          # latency, size, GPU memory per precision
btd evaluate --map                          # accuracy per precision on the BRISC test split
```

## TensorRT (optional)

```powershell
pip install tensorrt-cu12   # or tensorrt-cu13 — match torch.version.cuda (see `btd env`)
```

Then set `tensorrt.enabled: true` (`precision: fp16` or `int8`) in `configs/export.yaml` and re-run `btd export`.
`btd benchmark` automatically times any `*.engine` file in the model folder. Alternatively, keep using the ONNX
files and select `--providers tensorrt`, which uses ONNX Runtime's TensorRT execution provider and needs the
TensorRT libraries on `PATH`. Engines are cached next to the model.
