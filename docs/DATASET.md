# Dataset choice: BRISC 2025 instead of the Kaggle folder dataset

## TL;DR

| | Kaggle "Brain Tumor MRI" family (the local `Brain Tumor MRI Data/`) | **BRISC 2025** (used here) |
|---|---|---|
| Size | 7,200 slices (1,400 train + 400 test per class) in the local copy; the widely used Kaggle version has 7,023 | 6,000 slices (5,000 train / 1,000 test) |
| Classes | glioma, meningioma, pituitary, no tumour | same four |
| Labels | Folder name only | Folder labels, **pixel masks for all 4,793 tumour slices** (radiologist/physician reviewed), and imaging plane (axial/coronal/sagittal) |
| Sources | Merge of the figshare (Cheng et al.), SARTAJ and Br35H collections | Derived from the same public collections, then re-curated |
| Duplicates / leakage | Known problem: exact and near-duplicate slices, some across train/test | "Exact duplicates and near-duplicates … were removed; de-duplication was completed **before** any train/test split". The audit still finds train/test pairs, so `prepare` drops them from train (see below) |
| Integrity | None | `manifest.csv` with a SHA-256 per file; Zenodo publishes an MD5 for the archive |
| License | Varies by the upstream sources | **CC BY 4.0** |
| Supports | Classification only | Classification, **detection and instance segmentation** (boxes come from masks) |

Your current dataset can only train a classifier. It can't back a YOLO detection/segmentation claim, because it
has no boxes and no masks. Its test split is also unlikely to be independent of its training split.

## Check it yourself: the audit tool

```powershell
btd data audit --data "..\Brain Tumor MRI Data" --out reports/audit-kaggle
btd data audit --data data/raw --out reports/audit-brisc
```

For every image the tool computes:

- an MD5 of the decoded pixels, for **exact** duplicates (it catches re-saved files too)
- a 64-bit DCT perceptual hash, used to generate **near-duplicate candidates** within a Hamming radius of 10
- a 32×32 grayscale thumbnail. Each candidate pair is confirmed only when the Pearson correlation is ≥ 0.95, which
  keeps false positives low on MRI, where many slices share a dark background and a round skull

It then reports:

- class balance per split and the distribution of image sizes
- exact-duplicate groups and verified near-duplicate pairs (listed in `duplicate_pairs.csv`)
- **leakage**: the share of test images that have an exact or near copy in train, per class
- **label conflicts**: clusters of (near-)identical images filed under different classes

Paste `reports/audit-kaggle/audit.md` into your notes. It's concrete evidence for why accuracies of 99% or more on
that dataset should be distrusted.

## BRISC 2025 details

- 6,000 T1-weighted contrast-enhanced slices, 5,000 train and 1,000 test.
  Train: glioma 1,147 · meningioma 1,329 · pituitary 1,457 · no tumour 1,067.
  Test: glioma 254 · meningioma 306 · pituitary 300 · no tumour 140.
- Masks were drawn with AnyLabeling and reviewed by radiologists and physicians. On a quality-checked subset, initial
  and expert-verified masks agree with a mean Dice of 0.924.
- File naming: `brisc2025_<split>_<index>_<gl|me|pi|no>_<ax|co|sa>_<sequence>.jpg`, with the mask as a PNG under the
  same stem. The loader takes labels from the filename code, which is authoritative, and cross-checks them
  against folder names.
- Download: [Zenodo 10.5281/zenodo.17524350](https://doi.org/10.5281/zenodo.17524350) (`btd data download`
  verifies the MD5), or [Kaggle `briscdataset/brisc2025`](https://www.kaggle.com/datasets/briscdataset/brisc2025)
  (then run `btd data download --zip brisc2025.zip`).
- Paper: Fateh et al., *BRISC: Annotated Dataset for Brain Tumor Segmentation and Classification*, Scientific Data
  (2026), [doi:10.1038/s41597-026-06753-y](https://doi.org/10.1038/s41597-026-06753-y). The paper reports
  classification baselines up to about 0.99 weighted F1 (EfficientNet-B0) and segmentation baselines of roughly
  0.76–0.81 weighted mIoU. Compare with care: their IoU definition and pre-processing may differ from
  `btd evaluate`.

## How this repo splits the data

- **Test** = the official BRISC test split, untouched. It's used once, for the final report.
- **Train** = BRISC train minus every slice that is an exact or near duplicate of a test slice. Despite the release
  notes, the Zenodo release has such pairs: `btd data audit` found 101 of the 860 tumour test slices with a
  pixel-identical copy in train, and 402 with an exact or near copy. The dropped keys are listed in
  `prepare_report.json` under `train_slices_dropped_as_test_duplicates`.
- **Validation** = 15% of BRISC train, stratified by class and imaging plane, and **grouped by near-duplicate
  cluster**, so near-identical slices always stay on one side of the split. It's used for early stopping, choosing
  the head, and tuning the image-level threshold.
- No-tumour slices go into every split as **background images** (empty label files), so the detector learns not to
  fire on healthy anatomy.

The one limitation that remains is that BRISC ships no patient IDs, so a split by patient isn't possible. The
near-duplicate grouping reduces this risk but can't remove it entirely.
