# Nexus Bank AI API

KYC ML backend — the same Python, routes, and dependency stack as [`./kyc`](../kyc). AML screening is reserved and returns `501` until models are added.

Interactive docs: [Swagger UI](http://localhost:8000/docs) · [ReDoc](http://localhost:8000/redoc) · OpenAPI files in [`docs/`](docs/).

## Resources

Same contract as `kyc/api` (`ML_BACKEND_URL` in `kyc-api`):

| Method | Path | Domain |
|--------|------|--------|
| `GET` | `/health` | process liveness |
| `GET` | `/ready` | models loaded |
| `GET` | `/api/v1/health` | KYC model status |
| `POST` | `/api/v1/kyc/verify` | ID + selfie verification |
| `POST` | `/api/v1/kyc/ocr` | document OCR |
| `POST` | `/api/v1/ocr/extract` | document OCR (kyc path) |
| `GET` | `/api/v1/liveness/challenge` | create liveness challenge |
| `POST` | `/api/v1/liveness/verify` | verify challenge frames |
| `POST` | `/api/v1/liveness/detect` | batch liveness |
| `GET` | `/api/v1/aml/health` | AML placeholder |
| `POST` | `/api/v1/aml/screen` | AML placeholder (`501`) |

`/api/v1/kyc/liveness/*` aliases the liveness routes as well.

## Local run

```bash
python3.9 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_models.py   # fetches/verifies the InsightFace pack
export PYTHONPATH="$(pwd)"
uvicorn api.api:app --reload --port 8000
```

Open http://localhost:8000/docs.

Without ML weights (tests / schema only):

```bash
pip install -r requirements-dev.txt
export AI_API_SKIP_ML=1
pytest -q
python scripts/export_openapi.py
```

## Docker

```bash
docker compose up --build
```

The image follows `kyc/Dockerfile.ml-backend`: install the same Python stack, then download models during the build (YuNet via wget if missing, InsightFace from this repo's `weights-v1` release). On a native (non-QEMU) build it also runs `scripts/download_models.py --all` so PaddleOCR weights are cached and a model that cannot load fails the build instead of leaving the pod unready.

## ML weights

No weight is fetched from a third-party CDN at runtime. The small ones are committed; the InsightFace pack is published as a release asset on this repo, because at 166 MB it exceeds GitHub's hard 100 MB per-file limit for git.

| Path | Size | Source | Used by |
| --- | --- | --- | --- |
| `models/yunet.onnx` | 227 KB | committed | face detection |
| `models/haarcascade_*.xml` | 1.4 MB | committed | profile detection |
| `insightface/models/buffalo_l/w600k_r50.onnx` | 166 MB | `weights-v1` release | face matching (embeddings) |
| `insightface/models/buffalo_l/det_10g.onnx` | 16 MB | `weights-v1` release | unused, but `FaceAnalysis` asserts a detection model exists |

`buffalo_l` normally ships five models; the other three (`1k3d68`, `2d106det`, `genderage`) are not used here and are deliberately excluded.

### Publishing the weights release

Needed once, and again only if the pack changes:

```bash
gh release create weights-v1 --repo finetech-nexus/ai-api \
  --title 'InsightFace buffalo_l (trimmed)' \
  --notes 'det_10g.onnx and w600k_r50.onnx from the upstream v0.7 pack.' \
  insightface/models/buffalo_l/det_10g.onnx \
  insightface/models/buffalo_l/w600k_r50.onnx
```

Both locations come from `paths.*` in `configs/defaults.yaml`, resolved against this service root rather than the working directory or `$HOME`. `FaceAnalysis` defaults to `root='~/.insightface'` and downloads a fresh pack when it is missing, so a change to the runtime user would otherwise move the lookup and silently restore the network dependency at startup.

## Layout

```
api/              FastAPI app (kyc/api)
app/services/     YuNet, InsightFace, PaddleOCR, liveness (kyc/app)
configs/          defaults.yaml + env overlay (kyc/configs)
utils/            logging (kyc/utils)
models/           YuNet + Haar cascades
aml/              AML placeholder (501)
scripts/          download_models.py, export_openapi.py
```

## Kubernetes

`ai-api` is a subchart of `core-api-charts`. Enable it per overlay:

```yaml
ai-api:
  enabled: true
```

Point `kyc-api` `ML_BACKEND_URL` at the sibling service when you are ready to switch off `localhost:8000`.
