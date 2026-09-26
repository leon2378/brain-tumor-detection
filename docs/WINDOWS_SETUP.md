# Windows runbook: conda env `torch_new` and a 6 GB NVIDIA GPU

Run every command from **Anaconda Prompt** or **PowerShell**, inside the repo folder, with `torch_new` activated.

```text
Brain Tumor Detection\
└─ brain-tumor-detection\     ← this repo (run commands from here)
```

## 1. Install into the existing environment

```powershell
conda activate torch_new
cd "C:\Users\Leon-PC\Downloads\Projects\Brain Tumor Detection\brain-tumor-detection"
python -m pip install -e ".[train,serve,cpu,dev]"
btd env
```

`torch_new` needs Python 3.10 or newer (check with `python --version`). This project never installs or upgrades
**torch**. It uses whatever CUDA build is already in `torch_new`.
`btd env` prints what matters:

- `cuda_available: true`, `gpu`, `gpu_vram_gb` (about 6) and `torch_cuda` (for example `12.8`)
- `hints`, for example the exact `onnxruntime-gpu` version to install for GPU inference

If `cuda_available` is `false`, the torch in `torch_new` is a CPU build. Reinstall it with the command that
pytorch.org's "Get Started" selector gives for your CUDA version.

For **GPU inference and benchmarks** (optional; training doesn't need it), swap in the GPU build of ONNX Runtime
that matches torch's CUDA major version:

```powershell
pip uninstall -y onnxruntime
pip install "onnxruntime-gpu>=1.20,<1.27"   # torch built for CUDA 12.x (Python 3.11+; on 3.10 use <1.24)
# pip install "onnxruntime-gpu>=1.27"       # torch built for CUDA 13.x
```

## 2. Get BRISC 2025

```powershell
btd data download                         # Zenodo, ~260 MB + manifest, MD5 verified → data\raw
```

If Zenodo is slow or blocked, download `brisc2025.zip` in your browser from
[Zenodo](https://zenodo.org/records/17524350) or [Kaggle](https://www.kaggle.com/datasets/briscdataset/brisc2025),
then run:

```powershell
btd data download --zip "C:\Users\Leon-PC\Downloads\brisc2025.zip"
```

## 3. Audit the data (optional, 1–3 min)

```powershell
btd data audit --data data\raw --out reports\audit-brisc
```

Open `reports\audit-brisc\audit.md`. The leakage table shows what share of the test slices have a copy in train;
`btd data prepare` drops those from train. [DATASET.md](DATASET.md#audit-results) has the results for BRISC and
for the old Kaggle dataset.

## 4. Build the YOLO dataset

```powershell
btd data prepare                          # → data\processed\brisc-yolo (add --overwrite to rebuild)
```

This checks every file against its SHA-256 in `manifest.csv` (fetched by `btd data download`), converts masks to polygons (it reports how closely they
match), drops training slices that duplicate a test slice, and carves a leakage-safe validation split out of what's
left of BRISC train.

## 5. Train (6 GB profile)

```powershell
btd train                                        # yolo26s-seg @640, batch 8, AMP, early stopping
```

| Situation | Command |
|---|---|
| Quicker first run | `btd train model=yolo26n-seg.pt epochs=60 name=brisc-yolo26n` |
| `CUDA out of memory` | `btd train batch=4` (or `imgsz=512`) |
| DataLoader crash / "paging file is too small" | `btd train workers=2` |
| Resume after an interruption | `btd train model=runs\segment\brisc-yolo26s\weights\last.pt resume=True` |

Runs made before `btd train` pinned `project` to an absolute path sit one level deeper, under
`runs\segment\runs\segment\<name>\` — use that path when resuming or exporting from them.

The time for the first epoch, multiplied by the epoch count, gives a fair estimate; early stopping usually ends
sooner. Watch VRAM with `nvidia-smi -l 5`. When training finishes, `runs\segment\brisc-yolo26s\run_summary.json`
records the peak VRAM used — that's your evidence for the "fits in 6 GB" claim.

## 6. Export, quantise and tune the threshold

```powershell
btd export --device 0                     # picks the newest runs\segment\*\weights\best.pt
```

This scores both YOLO26 heads on val and keeps the better one. It then writes `artifacts\model\model_fp32.onnx`,
`model_fp16.onnx` and `model_int8.onnx`, plus `model.json` holding checksums and the threshold tuned on val.

## 7. Evaluate and benchmark on the test split

```powershell
btd evaluate --map                        # → reports\evaluation_test.md (+ confusion matrices)
btd benchmark --providers cpu cuda        # → reports\benchmark.md
```

Copy the numbers into the Results table in `README.md` and into `MODEL_CARD.md`.

## 8. Serve locally

```powershell
btd package --precision fp32              # → models\model.onnx + models\model.json
btd serve --model models\model.onnx       # web page at http://127.0.0.1:8000, API docs at /docs
```

Open http://127.0.0.1:8000 in your browser and try a sample slice, or drop in one of your own.

To call the API directly instead, use a second terminal (use `curl.exe`, because in PowerShell `curl` is an alias for `Invoke-WebRequest`):

```powershell
$img = Get-ChildItem data\processed\brisc-yolo\images\test\*_gl_*.jpg | Select-Object -First 1
curl.exe -F "file=@$($img.FullName)" http://127.0.0.1:8000/v1/predict
curl.exe -F "file=@$($img.FullName)" http://127.0.0.1:8000/v1/predict/overlay -o overlay.png
```

## 9. Push and let CI/CD run

```powershell
git init -b main
git add .
git commit -m "Rebuild: BRISC 2025, YOLO26-seg, INT8/FP16 ONNX, FastAPI, CI/CD"
git remote add origin https://github.com/leon2378/brain-tumor-detection.git
git push -u origin main
```

Data, runs, artefacts and weights are all git-ignored. To publish the trained model inside the Docker image, follow
"bake a trained model" in the README.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `'btd' is not recognized` | Activate `torch_new` again, or use `python -m btd …` |
| `CUDA out of memory` during training | `batch=4`, close other GPU apps, `imgsz=512`, or `model=yolo26n-seg.pt` |
| `LoadLibrary failed with error 126` or no `CUDAExecutionProvider` | The `onnxruntime-gpu` build doesn't match torch's CUDA major version. Follow the `btd env` hint |
| `FileExistsError: … is not empty` from `prepare` | Add `--overwrite` |
| Download stops part-way | Run `btd data download` again: it resumes from `.part` and re-verifies the MD5 |
| `Checksum mismatch` when serving | The model file was replaced or corrupted. Re-run `btd package` |
