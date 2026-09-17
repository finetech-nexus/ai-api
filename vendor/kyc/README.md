Vendored KYC ML engine, copied from `kyc/app`, `kyc/configs`, `kyc/utils`, and `kyc/models`.

The HTTP layer lives at the repo root (`main.py`, `api/`, `domains/`). This package keeps the original imports (`configs.config`, `app.services.*`, `utils.logger`) so detectors do not need a rewrite.

Set `PYTHONPATH` to include this directory (`vendor/kyc`).

## Local patches

This is no longer a byte-for-byte copy. Re-apply these when re-syncing from `kyc/`:

- `configs/config.py` — resolves `paths.*` against this directory instead of the process working directory. Without it, `models_dir: "models/"` resolves to `<cwd>/models` and the service starts but never becomes ready, because YuNet is not found.
- `configs/defaults.yaml` — adds `paths.insightface_root`.
- `app/services/face_matcher.py` — passes `root=paths.insightface_root` to `FaceAnalysis` so it reads the committed `insightface/models/buffalo_l/` rather than downloading a fresh pack into `~/.insightface`.

## Weights

- `models/yunet.onnx`, `models/haarcascade_*.xml` — face and profile detection.
- `insightface/models/buffalo_l/` — `w600k_r50.onnx` (embeddings) and `det_10g.onnx`. The three unused models from the upstream pack are omitted; `FaceAnalysis` loads every `.onnx` it finds, so they would cost image size and memory for nothing.
