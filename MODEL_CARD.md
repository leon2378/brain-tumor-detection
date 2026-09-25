# Model card — BRISC YOLO26-seg (brain tumour detection and segmentation)

## Model details

- **Architecture:** Ultralytics YOLO26 instance segmentation (default `yolo26s-seg`), fine-tuned from COCO-pretrained
  weights. It has one head for the classes, boxes and masks of three tumour types. `btd export` compares the
  NMS-free (one-to-one) head with the one-to-many + NMS head on validation and records its choice in
  `model.json → head_selection`.
- **Outputs:** per-instance class (glioma, meningioma, pituitary), confidence, box and mask. The image-level label
  (the three classes or `no_tumor`) comes from the top detection and an operating threshold tuned on validation.
- **Formats:** ONNX FP32 (reference), FP16 (GPU) and INT8 (CPU, static QDQ); optionally TensorRT. See
  [docs/QUANTIZATION.md](docs/QUANTIZATION.md).
- **Version and provenance:** `model.json` holds the version, creation time, git commit, training config summary,
  dataset DOI and SHA-256 of every artefact.
- **License:** AGPL-3.0 (inherited from Ultralytics). Dataset: CC BY 4.0.

## Intended use

- Portfolio, education and research on MRI tumour detection and segmentation, and on building MLOps pipelines.
- A reference implementation of a leakage-aware dataset workflow and of quantised, torch-free serving.

## Out of scope

- **Any clinical or diagnostic use**, triage, or use as a second reader. The model isn't a medical device and hasn't
  been validated prospectively or on external data.
- Sequences other than T1 contrast-enhanced, 3-D volumes, paediatric data, and tumour types outside the three classes.
- Anything that isn't a brain MRI slice. The API and web page accept any image and still return an answer, so it's up
  to the user to send only suitable images (see "Behaviour outside the training data" below).

## Training data

BRISC 2025 train split: 5,000 slices, minus 801 that are exact or near duplicates of a test slice (the audit found
101 test slices with a pixel-identical copy in train and 402 with an exact or near copy, so the release is not
de-duplicated across its own split). Of the 4,199 that remain, one tumour slice has no usable mask, and 15% is held
out as validation — stratified by class and plane, grouped by near-duplicate cluster. That leaves **3,568 train and
630 validation** slices. No-tumour slices are background images. See [docs/DATASET.md](docs/DATASET.md).

## Evaluation data and metrics

The official BRISC 2025 test split (1,000 slices) is used once, after every choice has been made on validation.

| Metric | FP32 | FP16 | INT8 |
|---|---|---|---|
| Box mAP50-95 | 0.6337 | 0.6338 | 0.6117 |
| Mask mAP50-95 | 0.6420 | 0.6423 | 0.6298 |
| Image-level accuracy (4 classes) | 0.9720 | 0.9720 | 0.9380 |
| Macro-F1 (4 classes) | 0.9724 | 0.9724 | 0.9342 |
| Tumour screening sensitivity / specificity | 0.9942 / 0.9857 | 0.9942 / 0.9857 | 0.9686 / 0.9857 |
| Mean Dice (tumour slices) | 0.8405 | 0.8410 | 0.8026 |
| Weighted dataset IoU | 0.7667 | 0.7676 | 0.7043 |
| p50 latency CPU / GPU (ms) | 74.0 / 14.0 | 76.9 / 12.4 | 47.5 / 28.5 |

Operating threshold 0.05, tuned on validation for macro-F1. Latency is end-to-end over 100 runs (RTX 3060 Laptop
6 GB; ONNX Runtime CUDA and CPU providers).

Per-class, image level (FP32): pituitary F1 0.980, no-tumour 0.975, glioma 0.970, meningioma 0.964. Of the 28
misclassified slices, 21 are tumour-type confusions, most often meningioma read as glioma (6) and pituitary as
meningioma (5). Five tumour slices are called healthy, and two healthy slices are called tumours.

Per-class mask mAP50-95 (FP32) is much less even: meningioma 0.782, pituitary 0.661, **glioma 0.483**. Gliomas have
irregular, diffuse borders, so boxes and masks are markedly weaker there than the image-level accuracy suggests.

These numbers are **in-distribution**: the test slices come from the same public collections as the training slices.
The model hasn't been evaluated on independent data, such as another hospital's scanners.

## Behaviour outside the training data

The model always answers with one of its four labels. It has no notion of "this isn't a brain MRI", and its
confidence says nothing about whether the input is suitable. A spot check with the released FP32 model
(September 2026):

| Input | Answer | Correct |
|---|---|---|
| BRISC glioma test slice, as trained | glioma 0.78 | yes |
| The same slice rotated 90° | glioma 0.56, with 7 scattered detections | partly |
| The same slice with inverted intensities (a stand-in for another MRI sequence) | meningioma 0.66 | no |
| The same slice heavily blurred | meningioma 0.06 | no |
| One slice per class from the Kaggle "Brain Tumor MRI" set | all four correct | yes, but not independent* |
| Random noise | no tumour | yes |
| A landscape photo (not an MRI at all) | **glioma 0.81**, 7 detections | no |

\* The Kaggle set is built from the same source collections as BRISC and shares many identical slices, so agreement
on it is no evidence of generalisation.

**Input check.** Because of the last row, the API adds a `not_greyscale` warning to its response when an image is
clearly in colour, and the web page shows it above the prediction. MRI slices are greyscale. The rule flags an image
when more than 10% of its pixels have colour channels that differ by more than 40 (on a 0–255 scale). That share is 0
for all 5,198 BRISC slices, at most 4.1% across the 7,200 Kaggle slices (some carry coloured annotations), and 27–98%
for photos. The check is a basic sanity test, not a guarantee: greyscale photos, other MRI sequences such as T2 or
FLAIR, and other body parts all pass it. Flagged requests are counted in the `btd_input_warnings_total` metric.

## Ethical considerations and limitations

- **False negatives matter most** in a screening framing. Report sensitivity alongside accuracy. The operating
  threshold maximises macro-F1, which isn't necessarily the right trade-off for any particular use.
- The source data is aggregated from public collections, with unknown demographics and scanners and a limited number
  of institutions. Expect degraded performance under domain shift.
- Slice-level splits: BRISC has no patient IDs, so leakage between slices of the same patient can't be completely
  ruled out, even after de-duplication and near-duplicate grouping.
- INT8 quantisation can shift decisions near the threshold. Here it costs 3.4 points of image-level accuracy
  (0.938 vs 0.972) and 2.6 points of screening sensitivity, in exchange for 1.6× faster CPU inference. Check the
  per-precision evaluation before deploying a quantised model.
