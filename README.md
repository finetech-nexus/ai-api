# Nexus Bank AI API

REST platform for Nexus Bank AI services. **KYC** (face match, document OCR, liveness) is implemented. **AML** (sanctions / PEP / transaction monitoring) is reserved and returns `501` until models are added.

Interactive docs: [Swagger UI](http://localhost:8000/docs) · [ReDoc](http://localhost:8000/redoc) · OpenAPI files in [`docs/`](docs/).

## Resources

| Method | Path | Domain |
|--------|------|--------|
| `GET` | `/health` | process liveness |
| `GET` | `/ready` | models loaded |
| `GET` | `/api/v1/health` | KYC model status |
| `POST` | `/api/v1/kyc/verify` | ID + selfie verification |
| `POST` | `/api/v1/kyc/ocr` | document OCR |
| `GET` | `/api/v1/kyc/liveness/challenge` | create liveness challenge |
| `POST` | `/api/v1/kyc/liveness/verify` | verify challenge frames |
| `POST` | `/api/v1/kyc/liveness/detect` | batch liveness |
| `GET` | `/api/v1/aml/health` | AML placeholder |
| `POST` | `/api/v1/aml/screen` | AML placeholder (`501`) |

Legacy aliases (same contract as `kyc/api`, used by `kyc-api` via `ML_BACKEND_URL`):

- `POST /api/v1/ocr/extract`
- `GET /api/v1/liveness/challenge`
- `POST /api/v1/liveness/verify`
- `POST /api/v1/liveness/detect`

## Local run

```bash
python3.9 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_models.py   # fetches/verifies the InsightFace pack
export PYTHONPATH="$(pwd):$(pwd)/vendor/kyc"
uvicorn main:app --reload --port 8000
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

The build runs `scripts/download_models.py --all`, which constructs every model once. Startup therefore needs no network, and a model that cannot load fails the build instead of leaving the pod permanently unready.

## ML weights

No weight is fetched from a third-party CDN. The small ones are committed; the InsightFace pack is published as a release asset on this repo, because at 166 MB it exceeds GitHub's hard 100 MB per-file limit for git.

| Path | Size | Source | Used by |
| --- | --- | --- | --- |
| `vendor/kyc/models/yunet.onnx` | 227 KB | committed | face detection |
| `vendor/kyc/models/haarcascade_*.xml` | 1.4 MB | committed | profile detection |
| `vendor/kyc/insightface/models/buffalo_l/w600k_r50.onnx` | 166 MB | `weights-v1` release | face matching (embeddings) |
| `vendor/kyc/insightface/models/buffalo_l/det_10g.onnx` | 16 MB | `weights-v1` release | unused, but `FaceAnalysis` asserts a detection model exists |

`buffalo_l` normally ships five models; the other three (`1k3d68`, `2d106det`, `genderage`) are not used here and are deliberately excluded. `FaceAnalysis` loads every `.onnx` it finds in the pack directory, so omitting them saves roughly 143 MB of image size and the same again in per-pod memory.

### Publishing the weights release

Needed once, and again only if the pack changes:

```bash
gh release create weights-v1 --repo finetech-nexus/ai-api \
  --title 'InsightFace buffalo_l (trimmed)' \
  --notes 'det_10g.onnx and w600k_r50.onnx from the upstream v0.7 pack.' \
  vendor/kyc/insightface/models/buffalo_l/det_10g.onnx \
  vendor/kyc/insightface/models/buffalo_l/w600k_r50.onnx
```

`scripts/download_models.py` fetches whatever is missing and verifies it against the SHA-256 digests pinned in the script, which are the upstream ones — so a swapped or truncated asset fails the build. Files already present and matching are left alone, which is why a local checkout with the weights never re-downloads. The release is public, so no credentials are involved; if the repo ever goes private, mirror the pack and point `AI_API_WEIGHTS_BASE_URL` at it.

Release assets allow 2 GB per file and do not consume Git LFS storage or bandwidth. Publishing a `weights-*` release does not trigger an image build — `release-image.yml` skips those tags.

Both locations come from `paths.*` in `vendor/kyc/configs/defaults.yaml`, resolved against `vendor/kyc` rather than the working directory or `$HOME`. That matters: `FaceAnalysis` defaults to `root='~/.insightface'` and downloads a fresh pack when it is missing, so a change to the runtime user (`securityContext.runAsUser`, a `USER` line) would otherwise move the lookup and silently restore the network dependency at startup. `scripts/download_models.py --all` asserts during the build that InsightFace resolved to the committed path.

PaddleOCR is the one exception: its detection and recognition weights are still fetched during the image build, because the cache location depends on the library version. `paddleocr` and `paddlepaddle` are pinned for that reason — see the comments in `requirements.txt`.

Image: `nexusbank/ai-api`. GitHub Actions build and push `latest` + git SHA on `main`, and version tags on `v*` releases.

Secrets for Actions: `DOCKER_USER`, `DOCKER_PASSWORD`. Optional variable: `DOCKER_ORGANIZATION` (default `nexusbank`).

## Layout

```
main.py           FastAPI entrypoint
api/              HTTP routers (KYC, AML, health)
domains/          KYC and AML business logic
vendor/kyc/       KYC ML engine (YuNet, InsightFace, PaddleOCR, liveness)
docs/             exported OpenAPI
.github/workflows CI, image build, release
```

New AI products (AML, fraud, …) go under `domains/<name>` and `api/v1/<name>`.

## Kubernetes

`ai-api` ships as a subchart of the `core-api` umbrella chart, alongside `core-banking-api` and `kyc-api`:

- Helm chart: `core-api-charts/ai-api`, published to `ghcr.io/finetech-nexus/core-api-charts`
- TN/dev values: `k8s-gitops/apps/api/overlays/dev/values.yaml` → `ai-api.api.dev.sandbox.internal.nbank.fr`
- EU/dev values: `k8s-gitops/apps/api-eu/overlays/dev/values.yaml` → `ai-api.api.dev1.sandbox.internal.nbank.fr`

It is gated behind `ai-api.enabled`, which defaults to `false` in the chart. An environment must opt in, and setting it back to `false` removes the workload without touching the sibling APIs — the escape hatch for sharing one release with them.

Bump `core-api-charts/Chart.yaml` and the `version` in the two `helmrelease-core-api.yaml` files together; Flux only pulls a chart version it is pinned to.

Since `ai-api` now runs in the same namespace as the Node BFF, point `kyc-api` `ML_BACKEND_URL` at the sibling service when you are ready to switch off `localhost:8000`:

```text
http://core-api-ai-api
```
