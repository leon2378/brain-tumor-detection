# Evaluation — brisc-yolo26s-seg on brisc-yolo (test)

| model | precision | split | acc | macro-F1 | sensitivity | specificity | Dice | IoU (dataset) | p50 ms |
|---|---|---|---|---|---|---|---|---|---|
| model_fp32.onnx | fp32 | test | 0.9720 | 0.9724 | 0.9942 | 0.9857 | 0.8405 | 0.7667 | 77.2 |
| model_fp16.onnx | fp16 | test | 0.9720 | 0.9724 | 0.9942 | 0.9857 | 0.8410 | 0.7676 | 82.4 |
| model_int8.onnx | int8 | test | 0.9380 | 0.9342 | 0.9686 | 0.9857 | 0.8026 | 0.7043 | 50.3 |

Confusion matrix (model_fp32.onnx, rows = truth):

| | glioma | meningioma | pituitary | no_tumor |
|---|---|---|---|---|
| **glioma** | 246 | 4 | 2 | 2 |
| **meningioma** | 6 | 293 | 4 | 3 |
| **pituitary** | 0 | 5 | 295 | 0 |
| **no_tumor** | 1 | 0 | 1 | 138 |
