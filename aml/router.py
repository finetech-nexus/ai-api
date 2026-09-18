"""AML REST resources. Not implemented yet — returns 501."""

from fastapi import APIRouter, status

from aml.schemas import (
    AmlHealthResponse,
    AmlScreenRequest,
    AmlScreenResponse,
    AmlScreenStatus,
)

router = APIRouter()


@router.get(
    "/health",
    response_model=AmlHealthResponse,
    summary="AML domain status",
)
async def aml_health():
    return AmlHealthResponse(
        message="AML models are not deployed yet. This resource is reserved.",
    )


@router.post(
    "/screen",
    response_model=AmlScreenResponse,
    status_code=status.HTTP_501_NOT_IMPLEMENTED,
    summary="Screen a party against watchlists",
    description="Placeholder. Will run sanctions, PEP, and adverse-media screening.",
)
async def screen_party(_body: AmlScreenRequest):
    return AmlScreenResponse(
        status=AmlScreenStatus.ERROR,
        message="AML screening is not implemented yet.",
        matches=[],
    )
