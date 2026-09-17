"""pytest configuration."""

import os

os.environ.setdefault("AI_API_SKIP_ML", "1")
os.environ.setdefault("ENV", "test")
