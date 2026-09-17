"""Versioned API router."""

from fastapi import APIRouter

from api.v1.aml.router import router as aml_router
from api.v1.kyc.router import router as kyc_router

api_router = APIRouter()
api_router.include_router(kyc_router, prefix="/kyc", tags=["KYC"])
api_router.include_router(aml_router, prefix="/aml", tags=["AML"])
