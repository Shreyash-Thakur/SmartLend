"""Customer profile endpoint.

Backs the read-only "we already have this" panel on the redesigned application
form: the applicant types their customer id and immediately sees the
demographic and bureau block the bank holds on them, rather than being asked to
re-key it (docs/FORM-REDESIGN.md).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.schemas import CustomerProfileResponse
from backend.app.services.customer_profile_service import get_profile

router = APIRouter(tags=["customers"])


@router.get("/customers/{customer_id}/profile", response_model=CustomerProfileResponse)
def read_customer_profile(customer_id: str) -> CustomerProfileResponse:
    profile = get_profile(customer_id)
    if profile is None:
        # 404, not an empty/synthetic profile: an unrecognised id must be
        # visibly unrecognised, never quietly filled in with plausible numbers.
        raise HTTPException(
            status_code=404,
            detail={
                "error": "Customer not found",
                "details": f"No customer profile on file for id {customer_id!r}.",
            },
        )
    return CustomerProfileResponse.model_validate(profile)
