"""Nexus Bank AI API — FastAPI application factory."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.v1.health import router as health_router
from api.v1.kyc.legacy import legacy_router
from api.v1.router import api_router
from core.errors import ErrorResponse
from core.logging import configure_logging
from core.settings import get_settings
from domains.kyc.runtime import runtime

OPENAPI_TAGS = [
    {"name": "Health", "description": "Liveness, readiness, and model status."},
    {
        "name": "KYC",
        "description": "Identity verification: face match, document OCR, and liveness.",
    },
    {
        "name": "AML",
        "description": "Anti-money-laundering screening (coming later).",
    },
]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await runtime.load()
    yield


def create_app() -> FastAPI:
    configure_logging()
    settings = get_settings()
    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Platform API for Nexus Bank AI services. "
            "KYC (face, OCR, liveness) is live. AML screening will be added as a separate domain."
        ),
        openapi_url=settings.openapi_url,
        docs_url=settings.docs_url,
        redoc_url=settings.redoc_url,
        lifespan=lifespan,
        openapi_tags=OPENAPI_TAGS,
        contact={"name": "Nexus Bank"},
        license_info={"name": "Proprietary"},
    )

    origins = settings.cors_origin_list()
    application.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=origins != ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(health_router)
    application.include_router(api_router, prefix="/api/v1")
    # Legacy aliases used by kyc-api (ML_BACKEND_URL): /api/v1/ocr/extract, /api/v1/liveness/*
    application.include_router(legacy_router, prefix="/api/v1", tags=["KYC"])

    @application.exception_handler(HTTPException)
    async def http_exception_handler(_request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=exc.__class__.__name__,
                message=str(exc.detail),
                details=None,
            ).model_dump(mode="json"),
        )

    @application.exception_handler(Exception)
    async def general_exception_handler(_request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="InternalServerError",
                message="An unexpected error occurred",
                details={"type": type(exc).__name__},
            ).model_dump(mode="json"),
        )

    return application


app = create_app()
