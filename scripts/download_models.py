#!/usr/bin/env python3
"""Check the ML weights the service needs, and optionally load them once.

All weights are committed under vendor/kyc, so this normally has nothing to
download. Run with --all during the image build to construct every model:

    python scripts/download_models.py --all

That proves the models load (a broken image fails the build instead of leaving
the pod unready) and caches PaddleOCR's weights, the one set still fetched from
the network.
"""

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "vendor" / "kyc")]

from configs.config import config  # noqa: E402

INSIGHTFACE_PACK = "buffalo_l"


def insightface_dir() -> Path:
    """Where FaceAnalysis(root=...) looks: <root>/models/<pack>."""
    root = Path(config.get("paths", "insightface_root", default="~/.insightface"))
    return root.expanduser() / "models" / INSIGHTFACE_PACK


def check_yunet() -> None:
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


def check_insightface() -> None:
    """The pack is committed; a download here would mean the path is wrong."""
    target = insightface_dir()
    models = sorted(target.glob("*.onnx"))
    if not models:
        raise SystemExit(
            "no committed InsightFace weights in {}; InsightFace would download "
            "the full pack instead".format(target)
        )
    for model in models:
        print("{} ({:.1f} MB)".format(model, model.stat().st_size / 1024 / 1024))


def warm_all() -> None:
    """Construct every model once."""
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
    built = {}
    for name, factory in loaders:
        print("Warming {}".format(name), flush=True)
        built[name] = factory()
        print("Warmed {}".format(name), flush=True)

    # Fail loudly if InsightFace ignored the committed pack and downloaded its
    # own copy, which would silently reintroduce the network dependency.
    resolved = Path(built["face_matcher"].app.model_dir).resolve()
    expected = insightface_dir().resolve()
    if resolved != expected:
        raise SystemExit(
            "InsightFace loaded {} instead of the committed {}".format(resolved, expected)
        )
    print("InsightFace used the committed weights at {}".format(resolved))


def main() -> int:
    parser = argparse.ArgumentParser(description="Check ML weights for ai-api.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="also construct every model, caching PaddleOCR's weights",
    )
    args = parser.parse_args()

    check_yunet()
    check_insightface()
    if args.all:
        warm_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
