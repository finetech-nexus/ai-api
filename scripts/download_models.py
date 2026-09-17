#!/usr/bin/env python3
"""Provision the ML weights the service needs.

Small weights are committed under vendor/kyc. The InsightFace pack is not: at
166 MB it exceeds GitHub's 100 MB per-file limit, so it is published as a release
asset on this repo and fetched here, verified against a pinned SHA-256.

    python scripts/download_models.py          # fetch what is missing
    python scripts/download_models.py --all    # also construct every model

Use --all in the image build: it caches PaddleOCR's weights and proves the models
load, so a broken image fails the build instead of leaving the pod unready.

The release is public, so no credentials are involved. Override the source with
AI_API_WEIGHTS_REPO, AI_API_WEIGHTS_TAG, or AI_API_WEIGHTS_BASE_URL.
"""

import argparse
import hashlib
import importlib
import os
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "vendor" / "kyc")]

from configs.config import config  # noqa: E402

WEIGHTS_REPO = os.environ.get("AI_API_WEIGHTS_REPO", "finetech-nexus/ai-api")
WEIGHTS_TAG = os.environ.get("AI_API_WEIGHTS_TAG", "weights-v1")

INSIGHTFACE_PACK = "buffalo_l"

# Trimmed buffalo_l: the recognition model plus the detection model FaceAnalysis
# asserts on. Digests are from the upstream v0.7 pack, so they hold wherever the
# asset is hosted. The other three models in the pack are unused here.
INSIGHTFACE_FILES = {
    "det_10g.onnx": "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
    "w600k_r50.onnx": "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def insightface_dir() -> Path:
    """Where FaceAnalysis(root=...) looks: <root>/models/<pack>."""
    root = Path(config.get("paths", "insightface_root", default="~/.insightface"))
    return root.expanduser() / "models" / INSIGHTFACE_PACK


def _urlopen(url, headers):
    request = urllib.request.Request(url, headers=headers)
    try:
        return urllib.request.urlopen(request)
    except urllib.error.HTTPError as exc:
        raise SystemExit(
            "HTTP {} ({}) fetching {}\n"
            "  If the weights release does not exist yet, publish it with:\n"
            "    gh release create {} --repo {} \\\n"
            "      --title 'InsightFace {} (trimmed)' \\\n"
            "      vendor/kyc/insightface/models/{}/*.onnx\n"
            "  A private repo cannot serve this URL: mirror the pack and set "
            "AI_API_WEIGHTS_BASE_URL.".format(
                exc.code, exc.reason, url,
                WEIGHTS_TAG, WEIGHTS_REPO,
                INSIGHTFACE_PACK, INSIGHTFACE_PACK,
            )
        )


def _asset_url(name: str) -> str:
    """Public release asset URL.

    The repo is public, so no authentication is involved. Point
    AI_API_WEIGHTS_BASE_URL elsewhere to serve the pack from an internal mirror.
    """
    base = os.environ.get(
        "AI_API_WEIGHTS_BASE_URL",
        "https://github.com/{}/releases/download/{}".format(WEIGHTS_REPO, WEIGHTS_TAG),
    )
    return "{}/{}".format(base.rstrip("/"), name)


def _download(url, target: Path, expected: str) -> None:
    print("Downloading {} -> {}".format(url, target), flush=True)
    tmp = target.with_suffix(target.suffix + ".part")
    with _urlopen(url, {}) as response, open(tmp, "wb") as handle:
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            handle.write(block)

    actual = sha256(tmp)
    if actual != expected:
        tmp.unlink()
        raise SystemExit(
            "checksum mismatch for {}\n  expected {}\n  got      {}".format(
                target.name, expected, actual
            )
        )
    tmp.replace(target)
    print("Wrote {} ({:.1f} MB)".format(target, target.stat().st_size / 1024 / 1024))


def fetch_yunet() -> None:
    """Committed under vendor/kyc/models; fetch only if somehow absent."""
    models_dir = Path(config.get("paths", "models_dir", default="models"))
    models_dir.mkdir(parents=True, exist_ok=True)
    target = models_dir / config.get(
        "models", "face_detection", "local_file", default="yunet.onnx"
    )

    if target.exists():
        print("{} ({:.1f} MB)".format(target, target.stat().st_size / 1024 / 1024))
        return

    url = config.get("models", "face_detection", "url")
    print("{} missing, downloading {}".format(target, url))
    tmp = target.with_suffix(target.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(target)


def fetch_insightface() -> None:
    dest = insightface_dir()
    dest.mkdir(parents=True, exist_ok=True)

    missing = {}
    for name, expected in INSIGHTFACE_FILES.items():
        target = dest / name
        if target.exists() and sha256(target) == expected:
            print("{} ({:.1f} MB, verified)".format(target, target.stat().st_size / 1024 / 1024))
        else:
            missing[name] = expected
    if not missing:
        return

    for name, expected in missing.items():
        _download(_asset_url(name), dest / name, expected)


LOADERS = (
    ("face_detector", "app.services.face_detector_id", "get_face_detector"),
    ("face_matcher", "app.services.face_matcher", "get_face_matcher"),
    ("ocr_extractor", "app.services.ocr_extractor", "get_ocr_extractor"),
    ("liveness_detector", "app.services.liveness_detector", "get_liveness_detector"),
)


def warm_all() -> None:
    """Construct every model.

    Every loader is attempted even after one fails, so a single build reports the
    state of all four rather than only the first problem. Imports happen per model
    so an import error is attributed to the model that caused it.
    """
    built = {}
    failures = []
    for name, module_name, factory_name in LOADERS:
        print("=== warming {}".format(name), flush=True)
        started = time.monotonic()
        try:
            factory = getattr(importlib.import_module(module_name), factory_name)
            built[name] = factory()
        except Exception:
            failures.append(name)
            print("=== FAILED {}".format(name), flush=True)
            traceback.print_exc()
            sys.stderr.flush()
        else:
            print(
                "=== ok {} in {:.1f}s".format(name, time.monotonic() - started),
                flush=True,
            )

    if "face_matcher" in built:
        # Fail loudly if InsightFace ignored the provisioned pack and downloaded
        # its own copy from upstream, which would leave 143 MB of unused models in
        # the image and reintroduce the third-party dependency.
        resolved = Path(built["face_matcher"].app.model_dir).resolve()
        expected = insightface_dir().resolve()
        if resolved != expected:
            raise SystemExit("InsightFace loaded {} instead of {}".format(resolved, expected))
        print("InsightFace used the provisioned weights at {}".format(resolved))

    print("\nsummary: {} loaded, {} failed".format(len(built), len(failures)))
    for name, _, _ in LOADERS:
        print("  {:<18} {}".format(name, "ok" if name in built else "FAILED"))
    if failures:
        raise SystemExit("could not construct: {}".format(", ".join(failures)))


def main() -> int:
    parser = argparse.ArgumentParser(description="Provision ML weights for ai-api.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="also construct every model, caching PaddleOCR's weights",
    )
    args = parser.parse_args()

    fetch_yunet()
    fetch_insightface()
    if args.all:
        warm_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
