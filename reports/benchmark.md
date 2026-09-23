# Inference benchmark — brisc-yolo26s-seg

100 timed requests per row, end-to-end (pre-process + inference + decode + masks).

| precision | provider | size (MB) | p50 ms | p95 ms | inference p50 ms | img/s | GPU mem (MB) | status |
|---|---|---|---|---|---|---|---|---|
| fp32 | cpu | 41.78 | 74.01 | 95.0 | 71.24 | 12.94 |  | ok |
| fp32 | cuda | 41.78 | 13.97 | 21.39 | 11.34 | 65.37 | 376.7 | ok |
| fp16 | cpu | 20.96 | 76.93 | 95.09 | 74.05 | 12.61 |  | ok |
| fp16 | cuda | 20.96 | 12.43 | 23.37 | 9.87 | 68.99 | 182.0 | ok |
| int8 | cpu | 11.34 | 47.47 | 53.0 | 44.5 | 20.57 |  | ok |
| int8 | cuda | 11.34 | 28.48 | 55.69 | 25.88 | 27.4 | 368.0 | ok |
