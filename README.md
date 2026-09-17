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

GitOps lives in `k8s-gitops`:

- Helm chart: `apps/ai-api/chart`
- TN/dev: `apps/ai-api/overlays/dev` → `ai-api.api.dev.sandbox.internal.nbank.fr`
- EU/dev: `apps/ai-api-eu/overlays/dev` → `ai-api.api.dev1.sandbox.internal.nbank.fr`

After the image is in Docker Hub, Flux reconciles the HelmRelease. Point `kyc-api` `ML_BACKEND_URL` at the in-cluster service when you are ready to switch the Node BFF off `localhost:8000`:

```text
http://ai-api.ai.svc.cluster.local
```
