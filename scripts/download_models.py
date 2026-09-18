#!/usr/bin/env python3
"""Provision the ML weights the service needs.

Small weights are committed under models/. The InsightFace pack is not: at
166 MB it exceeds GitHub's 100 MB per-file limit, so it is published as a
release asset on this repo and fetched here, verified against a pinned SHA-256.

    python scripts/download_models.py
        Fetch missing weights.

    python scripts/download_models.py --all
        Fetch all weights and warm every ML model.

    python scripts/download_models.py --warm
        Warm every ML model without downloading weights.

ARM64 builds should normally fetch the weights without --all. Native model
initialization is intentionally not performed during ARM64 Docker builds because
some native ML runtimes can segfault during build-time execution.

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
KYC_ROOT = ROOT
sys.path[:0] = [str(ROOT)]


WEIGHTS_REPO = os.environ.get(
    "AI_API_WEIGHTS_REPO",
    "finetech-nexus/ai-api",
)

WEIGHTS_TAG = os.environ.get(
    "AI_API_WEIGHTS_TAG",
    "weights-v1",
)

INSIGHTFACE_PACK = "buffalo_l"

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)


# Only the models actually required by this service.
INSIGHTFACE_FILES = {
    "det_10g.onnx":
        "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",

    "w600k_r50.onnx":
        "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def insightface_dir() -> Path:
    """Return the directory used by FaceAnalysis(root=...)."""
    return KYC_ROOT / "insightface" / "models" / INSIGHTFACE_PACK


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
            "      insightface/models/{}/*.onnx\n"
            "  A private repo cannot serve this URL: mirror the pack and set "
            "AI_API_WEIGHTS_BASE_URL.".format(
                exc.code,
                exc.reason,
                url,
                WEIGHTS_TAG,
                WEIGHTS_REPO,
                INSIGHTFACE_PACK,
                INSIGHTFACE_PACK,
            )
        )


def _asset_url(name: str) -> str:
    base = os.environ.get(
        "AI_API_WEIGHTS_BASE_URL",
        "https://github.com/{}/releases/download/{}".format(
            WEIGHTS_REPO,
            WEIGHTS_TAG,
        ),
    )

    return "{}/{}".format(
        base.rstrip("/"),
        name,
    )


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
            "checksum mismatch for {}\n"
            "  expected {}\n"
            "  got      {}".format(
                target.name,
                expected,
                actual,
            )
        )

    tmp.replace(target)

    print(
        "Wrote {} ({:.1f} MB)".format(
            target,
            target.stat().st_size / 1024 / 1024,
        ),
        flush=True,
    )


def fetch_yunet() -> None:
    """Fetch YuNet if it is not already present."""

    models_dir = KYC_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    target = models_dir / "yunet.onnx"

    if target.exists():
        print(
            "{} ({:.1f} MB)".format(
                target,
                target.stat().st_size / 1024 / 1024,
            ),
            flush=True,
        )
        return

    print(
        "{} missing, downloading {}".format(
            target,
            YUNET_URL,
        ),
        flush=True,
    )

    tmp = target.with_suffix(target.suffix + ".part")

    urllib.request.urlretrieve(
        YUNET_URL,
        tmp,
    )

    tmp.replace(target)


def fetch_insightface() -> None:
    """Fetch the trimmed InsightFace buffalo_l pack."""

    dest = insightface_dir()
    dest.mkdir(parents=True, exist_ok=True)

    missing = {}

    for name, expected in INSIGHTFACE_FILES.items():
        target = dest / name

        if target.exists() and sha256(target) == expected:
            print(
                "{} ({:.1f} MB, verified)".format(
                    target,
                    target.stat().st_size / 1024 / 1024,
                ),
                flush=True,
            )
        else:
            missing[name] = expected

    if not missing:
        return

    for name, expected in missing.items():
        _download(
            _asset_url(name),
            dest / name,
            expected,
        )


LOADERS = (
    (
        "face_detector",
        "app.services.face_detector_id",
        "get_face_detector",
    ),
    (
        "face_matcher",
        "app.services.face_matcher",
        "get_face_matcher",
    ),
    (
        "ocr_extractor",
        "app.services.ocr_extractor",
        "get_ocr_extractor",
    ),
    (
        "liveness_detector",
        "app.services.liveness_detector",
        "get_liveness_detector",
    ),
)


def warm_all() -> None:
    """Construct every model.

    This intentionally executes native ML runtimes and should therefore only
    be used when the build environment is known to support them.
    """

    built = {}
    failures = []

    for name, module_name, factory_name in LOADERS:
        print(
            "=== warming {}".format(name),
            flush=True,
        )

        started = time.monotonic()

        try:
            factory = getattr(
                importlib.import_module(module_name),
                factory_name,
            )

            built[name] = factory()

        except Exception:
            failures.append(name)

            print(
                "=== FAILED {}".format(name),
                flush=True,
            )

            traceback.print_exc()
            sys.stderr.flush()

        else:
            print(
                "=== ok {} in {:.1f}s".format(
                    name,
                    time.monotonic() - started,
                ),
                flush=True,
            )

    if "face_matcher" in built:
        resolved = Path(
            built["face_matcher"].app.model_dir
        ).resolve()

        expected = insightface_dir().resolve()

        if resolved != expected:
            raise SystemExit(
                "InsightFace loaded {} instead of {}".format(
                    resolved,
                    expected,
                )
            )

        print(
            "InsightFace used the provisioned weights at {}".format(
                resolved
            ),
            flush=True,
        )

    print(
        "\nsummary: {} loaded, {} failed".format(
            len(built),
            len(failures),
        ),
        flush=True,
    )

    for name, _, _ in LOADERS:
        print(
            "  {:<18} {}".format(
                name,
                "ok" if name in built else "FAILED",
            ),
            flush=True,
        )

    if failures:
        raise SystemExit(
            "could not construct: {}".format(
                ", ".join(failures)
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Provision ML weights for ai-api."
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="download weights and warm every model",
    )

    parser.add_argument(
        "--warm",
        action="store_true",
        help="warm every model without downloading weights",
    )

    args = parser.parse_args()

    # Always make sure the required weights exist.
    if not args.warm:
        fetch_yunet()
        fetch_insightface()

    # Explicitly requested native model initialization.
    if args.all or args.warm:
        warm_all()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
