"""Image decode helpers used by KYC (and later other vision domains)."""

from __future__ import annotations

import base64
from io import BytesIO

from fastapi import HTTPException, UploadFile, status

from core.settings import get_settings


async def read_upload_file(upload_file: UploadFile):
    """Read and validate an uploaded image, resizing oversized payloads."""
    import cv2
    import numpy as np

    settings = get_settings()
    max_size = settings.max_upload_size_mb * 1024 * 1024
    content = await upload_file.read()

    try:
        nparr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid image format. Supported: JPG, PNG",
            )

        max_dim = settings.image_max_dimension
        h, w = image.shape[:2]
        needs_resize = h > max_dim or w > max_dim or len(content) > max_size
        if needs_resize:
            scale = 1.0
            if h > max_dim or w > max_dim:
                scale = min(scale, max_dim / float(max(h, w)))
            if len(content) > max_size:
                scale = min(scale, 0.5)
            image = cv2.resize(
                image,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )

        min_width, min_height = 320, 240
        h, w = image.shape[:2]
        if w < min_width or h < min_height:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Image resolution too low ({w}x{h}). "
                    f"Please upload a higher resolution image (minimum {min_width}x{min_height})."
                ),
            )

        return image
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to decode image",
        ) from exc


def decode_base64_image(base64_str: str):
    """Decode a base64 image (with or without data URI prefix) to BGR numpy array."""
    import cv2
    import numpy as np
    from PIL import Image

    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[1]

    try:
        image_data = base64.b64decode(base64_str)
        pil_image = Image.open(BytesIO(image_data))
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        img_array = np.array(pil_image)
        return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to decode base64 image: {exc}",
        ) from exc
