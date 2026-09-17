#!/usr/bin/env python3
"""Download and cache the ML weights the service needs.

Weights are not committed to the repo. Run with --all during the image build so
the container never has to reach the internet at startup:

    python scripts/download_models.py --all

Without --all only YuNet is fetched, which is all a local checkout needs before
the heavy ML wheels are installed.
"""

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "vendor" / "kyc")]

from configs.config import config  # noqa: E402


def download_yunet() -> None:
    models_dir = Path(config.get("paths", "models_dir", default="models"))
    models_dir.mkdir(parents=True, exist_ok=True)

    url = config.get("models", "face_detection", "url")
    target = models_dir / config.get(
        "models", "face_detection", "local_file", default="yunet.onnx"
    )

    if target.exists():
        print("{} already present".format(target))
        return

    print("Downloading {} -> {}".format(url, target))
    tmp = target.with_suffix(target.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(target)
    print("Wrote {} ({:.1f} MB)".format(target, target.stat().st_size / 1024 / 1024))


def warm_all() -> None:
    """Construct every model once.

    InsightFace and PaddleOCR fetch their weight packs on first construction and
    cache them under $HOME, so doing this at build time is what keeps startup
    offline. It also means a model that cannot load fails the build instead of
    leaving the pod permanently unready.
    """
    from app.services.face_detector_id import get_face_detector
    from app.services.face_matcher import get_face_matcher
    from app.services.ocr_extractor import get_ocr_extractor
    from app.services.liveness_detector import get_liveness_detector

    loaders = (
        ("face_detector", get_face_detector),
        ("face_matcher", get_face_matcher),
        ("ocr_extractor", get_ocr_extractor),
        ("liveness_detector", get_liveness_detector),
    )
    for name, factory in loaders:
        print("Warming {}".format(name), flush=True)
        factory()
        print("Warmed {}".format(name), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch ML weights for ai-api.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="also cache the InsightFace and PaddleOCR weight packs",
    )
    args = parser.parse_args()

    download_yunet()
    if args.all:
        warm_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
