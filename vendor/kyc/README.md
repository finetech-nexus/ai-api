Vendored KYC ML engine, copied from `kyc/app`, `kyc/configs`, `kyc/utils`, and `kyc/models`.

The HTTP layer lives at the repo root (`main.py`, `api/`, `domains/`). This package keeps the original imports (`configs.config`, `app.services.*`, `utils.logger`) so detectors do not need a rewrite.

Set `PYTHONPATH` to include this directory (`vendor/kyc`).
