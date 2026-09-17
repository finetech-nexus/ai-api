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
python scripts/download_models.py   # fetches yunet.onnx into vendor/kyc/models
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

The build runs `scripts/download_models.py --all`, which fetches YuNet and constructs the InsightFace and PaddleOCR models once so their weight packs are cached in the image. Startup therefore needs no network, and a model that cannot load fails the build instead of leaving the pod permanently unready. The trade-off is a slower, larger build.

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
