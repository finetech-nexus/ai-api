"""Export OpenAPI JSON/YAML from the running app schema (no server required)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api.api import app  # noqa: E402


def main() -> None:
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    spec = app.openapi()
    (docs / "openapi.json").write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    (docs / "openapi.yaml").write_text(
        yaml.safe_dump(spec, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"Wrote {docs / 'openapi.json'} and {docs / 'openapi.yaml'}")


if __name__ == "__main__":
    main()
