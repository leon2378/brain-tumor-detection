# models/

This folder is what the Docker image serves (`/app/models`). It stays empty in git.

```powershell
btd package --precision int8               # → models/model.onnx + models/model.json (CPU image)
btd package --precision fp16 --dest models-gpu   # GPU image (docker compose --profile gpu)
```

`model.json` pins the SHA-256 of `model.onnx`. The server refuses to load a file that doesn't match, so always copy
the pair together. When the folder is empty, the API starts but `/ready` returns 503 until a model is mounted.
