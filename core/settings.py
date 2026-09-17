"""Runtime settings for the AI API."""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Nexus Bank AI API"
    app_version: str = "0.1.0"
    environment: str = Field(default="development", alias="ENV")
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    cors_origins: str = "*"
    api_key: Optional[str] = None
    max_upload_size_mb: int = 10
    image_max_dimension: int = 4096
    max_concurrent_requests: int = 10
    AI_API_SKIP_ML: bool = False

    docs_url: str = "/docs"
    redoc_url: str = "/redoc"
    openapi_url: str = "/openapi.json"

    def cors_origin_list(self) -> List[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
