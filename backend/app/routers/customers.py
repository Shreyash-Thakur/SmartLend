"""Customer profile endpoint.

Backs the read-only "we already have this" panel on the redesigned application
form: the applicant types their customer id and immediately sees the
demographic and bureau block the bank holds on them, rather than being asked to
re-key it (docs/FORM-REDESIGN.md).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.schemas import CustomerProfileResponse, CustomerSampleResponse
from backend.app.services.customer_profile_service import get_profile, get_sample_customers

router = APIRouter(tags=["customers"])


# Declared BEFORE `/customers/{customer_id}/profile` only for readability; the
# two paths cannot collide (different shapes), so order is not load-bearing.
@router.get("/customers/samples", response_model=list[CustomerSampleResponse])
def read_customer_samples(limit: int = 10) -> list[dict]:
    """Example customer ids the form can offer as "try one of these".

    Without this the applicant has to already know a valid `SK_ID_CURR`, which
    nobody does on a fresh clone. Returns [] rather than 404 when nothing is
    seeded, so the UI can simply hide the panel.
    """
    return get_sample_customers(limit=max(1, min(limit, 50)))


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
