from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ManualDecisionRequest(BaseModel):
    status: str
    notes: str = ""

    # --- relearning-loop reviewer capture (spec section 3) -----------------
    # All optional so existing clients keep working unchanged; when a field is
    # absent the corresponding column is simply left NULL rather than guessed.
    # `reviewerConfidence` is the 1-5 self-rating used to model reviewer
    # consistency (Madras et al.); `timeSpentSeconds` feeds the
    # reviewer-attention-degradation check. None of these are ever read as a
    # training label — see docs/RELEARNING-LOOP.md.
    reviewerId: str | None = None
    reviewerConfidence: int | None = Field(default=None, ge=1, le=5)
    timeSpentSeconds: float | None = Field(default=None, ge=0)
    reasonCodes: list[str] | None = None


# Short-form <-> legacy field-name bridge. The redesigned form speaks
# snake_case (docs/FORM-REDESIGN.md); every existing caller, stored row and
# response builder speaks camelCase. Both directions are filled in so neither
# side has to know about the other.
_SHORT_TO_LEGACY: dict[str, str] = {
    "loan_amount": "loanAmount",
    "loan_purpose": "loanPurpose",
    "loan_tenure_months": "loanTenure",
}

# A guarantor annexure is demanded only for unsecured borrowing above this
# amount, mirroring real lender forms where the guarantor block is a separate
# annexure rather than an always-on section.
GUARANTOR_THRESHOLD = 500_000.0


class LoanApplicationInput(BaseModel):
    """The redesigned short application form (docs/FORM-REDESIGN.md).

    The applicant is an existing bank customer, so the form asks only for what
    the bank cannot answer itself: 14 fields across three sections, plus the
    `customer_id` that resolves the rest.

    Everything the applicant used to type — age, gender, region, income,
    employment, and above all `cibilScore` (a bureau pull, never self-reported)
    — is now filled in by `customer_profile_service.resolve_application_payload()`
    before validation.

    BACKWARD COMPATIBILITY
    ----------------------
    Those legacy fields are still *declared* here, and still appear in
    `model_dump()`, because `build_application_response()`, the stored
    `input_data` blobs, the seeded rows and the org dashboard all read them by
    their camelCase names. What changed is that none of them is required any
    more: `loanAmount`, `monthlyIncome` and `age` used to be mandatory inputs
    and are now optional, filled either by the profile lookup or by a legacy
    caller that still posts them. Their validators fire only when a value is
    actually present, so an old-vocabulary payload behaves exactly as before
    while a short-form payload no longer has to invent values to get past them.
    """

    model_config = ConfigDict(extra="allow")

    # ---- Identity of the existing customer -------------------------------
    customer_id: str | None = None

    # ---- Section A · Loan request (6 fields) -----------------------------
    loan_amount: float | None = None
    loan_purpose: str = "personal"
    loan_tenure_months: int = 36
    repayment_source: str = "salary"
    preferred_emi_date: int = 5
    end_use_declaration: bool = False  # RBI: no speculative / anti-social use

    # ---- Section B · What the bank cannot see (5 fields) ------------------
    obligations_other_banks: float = 0.0
    employment_changed_recently: bool = False
    additional_income: float = 0.0
    collateral_type: str = "none"
    collateral_value: float = 0.0
    is_related_party: bool = False  # RBI: bank director or relative of one

    # ---- Section C · Guarantor (3 fields, conditional) --------------------
    guarantor_required: bool = False  # auto-computed, not user input
    guarantor_relationship: str | None = None
    guarantor_customer_id: str | None = None

    # ---- Legacy vocabulary (backward compatibility, not asked of the user) --
    # Populated from the customer profile, or by an old-style caller. Kept so
    # existing readers of `input_data` keep working unchanged.
    firstName: str = "Customer"
    lastName: str = "Applicant"
    email: str = "customer@example.com"
    phone: str = "+91 9000000000"
    loanAmount: float | None = None
    loanPurpose: str = "personal"
    loanTenure: int = 36
    interestRate: float = 12.0
    monthlyIncome: float | None = None
    annualIncome: float | None = None
    emi: float = 0.0
    existingEmis: float = 0.0
    residentialAssetsValue: float = 0.0
    commercialAssetsValue: float = 0.0
    bankBalance: float = 0.0
    cibilScore: int | None = None
    totalLoans: int = 0
    activeLoans: int = 0
    closedLoans: int = 0
    missedPayments: int = 0
    creditUtilizationRatio: float = 0.0
    age: int | None = None
    dependents: int = 0
    employmentType: str = "salaried"
    yearsOfEmployment: int = 0
    # `region` is deliberately untyped: CBES scores it as a number
    # (REGION_RATING_CLIENT, 1=best..3=worst) while older callers and the
    # geography report pass a label like "west". Coercing either way would
    # corrupt the other, so both are accepted verbatim.
    region: Any = "west"
    city: str = "Unknown"
    gender: str = "other"
    maritalStatus: str = "single"

    @model_validator(mode="before")
    @classmethod
    def sync_short_and_legacy_names(cls, data: Any) -> Any:
        """Accept either vocabulary on input and mirror it onto the other."""
        if not isinstance(data, dict):
            return data
        payload = dict(data)
        for short_key, legacy_key in _SHORT_TO_LEGACY.items():
            if payload.get(short_key) is None and payload.get(legacy_key) is not None:
                payload[short_key] = payload[legacy_key]
            elif payload.get(legacy_key) is None and payload.get(short_key) is not None:
                payload[legacy_key] = payload[short_key]
        return payload

    @field_validator("age")
    @classmethod
    def validate_age(cls, value: int | None) -> int | None:
        # Only enforced when supplied; a short-form submission gets age from
        # the profile lookup, which has already been merged in by this point.
        if value is not None and (value < 18 or value > 70):
            raise ValueError("age must be between 18 and 70")
        return value

    @field_validator("monthlyIncome")
    @classmethod
    def validate_income(cls, value: float | None) -> float | None:
        if value is not None and value <= 0:
            raise ValueError("monthlyIncome must be greater than 0")
        return value

    @field_validator("loanAmount", "loan_amount")
    @classmethod
    def validate_loan_amount(cls, value: float | None) -> float | None:
        if value is not None and value <= 0:
            raise ValueError("loanAmount must be greater than 0")
        return value

    @field_validator("cibilScore")
    @classmethod
    def validate_cibil(cls, value: int | None) -> int | None:
        if value is not None and (value < 300 or value > 900):
            raise ValueError("cibilScore must be between 300 and 900")
        return value

    @model_validator(mode="after")
    def finalize(self) -> "LoanApplicationInput":
        if self.loan_amount is None and self.loanAmount is None:
            raise ValueError("loan_amount is required")

        # Keep the two vocabularies in lockstep after coercion.
        if self.loan_amount is None:
            self.loan_amount = self.loanAmount
        if self.loanAmount is None:
            self.loanAmount = self.loan_amount
        self.loanPurpose = self.loan_purpose or self.loanPurpose
        self.loan_purpose = self.loanPurpose
        self.loanTenure = self.loan_tenure_months or self.loanTenure
        self.loan_tenure_months = self.loanTenure

        if self.emi and self.monthlyIncome is not None and self.emi > self.monthlyIncome:
            raise ValueError("emi must be less than or equal to monthlyIncome")
        if self.annualIncome is None and self.monthlyIncome is not None:
            self.annualIncome = float(self.monthlyIncome) * 12

        # EMIs the bank cannot see are real obligations; fold them into the
        # legacy `existingEmis` field the scoring path already reads.
        if self.obligations_other_banks:
            self.existingEmis = float(self.existingEmis) + float(self.obligations_other_banks)

        # `guarantor_required` is derived, never trusted from the request.
        amount = float(self.loan_amount or 0.0)
        self.guarantor_required = (
            self.collateral_type in ("", "none", None) and amount > GUARANTOR_THRESHOLD
        )
        return self


class CustomerProfileResponse(BaseModel):
    """The read-only "we already have this" block, keyed by customer id."""

    model_config = ConfigDict(extra="allow")

    customer_id: str
    found: bool = True
    age: int | None = None
    gender: str | None = None
    dependents: int | None = None
    marital_status: str | None = None
    region: str | None = None
    region_rating: int | None = None
    employment_type: str | None = None
    employment_tenure_years: float | None = None
    annual_income: float | None = None
    monthly_income: float | None = None
    existing_emis: float | None = None
    dti: float | None = None
    credit_score: float | None = None
    credit_score_display: int | None = None
    credit_score_basis: str | None = None
    delinquencies: int | None = None
    active_loans: int | None = None
    closed_loans: int | None = None
    total_loans: int | None = None
    credit_utilisation: float | None = None


class CustomerSampleResponse(BaseModel):
    """One example customer id for the form's "try one of these" panel."""

    model_config = ConfigDict(extra="allow")

    customer_id: str
    descriptor: str
    # A hint from the profile's own drivers, NOT an engine prediction: the real
    # decision also needs the loan amount, which does not exist until the form
    # is filled in.
    expected_decision_hint: str | None = None
    credit_score_display: int | None = None
    annual_income: float | None = None


class DecisionPayload(BaseModel):
    id: str
    status: str
    decidedAt: datetime
    decidedBy: str
    riskScore: float
    cbessScore: float
    uncertainty: float
    confidence: str
    explanation: str
    positiveFactors: list[str]
    negativeFactors: list[str]
    featureImportance: list[dict[str, Any]]
    modelVersion: str
    analystId: str | None = None
    analystNotes: str | None = None


class LoanApplicationResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    createdAt: datetime
    updatedAt: datetime
    status: str
    source: str
    applicantId: str
    applicantName: str
    email: str
    phone: str
    loanAmount: float
    loanPurpose: str
    loanTenure: int
    interestRate: float | None = None
    ml_prob: float | None = None
    cbes_prob: float | None = None
    cbes_score: float | None = None
    confidence: float | None = None
    finalDecision: str | None = None
    applicationData: dict[str, Any]
    decision: DecisionPayload | None = None
    documents: list[dict[str, Any]] = Field(default_factory=list)


class DocumentUploadResponse(BaseModel):
    fileName: str
    documentType: str
    uploadedAt: datetime
    extractedData: dict[str, Any] | None = None
    mappedData: dict[str, Any] | None = None
    fileSize: int


class ApplicationExplainResponse(BaseModel):
    id: str
    decision: str
    topFactors: list[dict[str, Any]] = Field(default_factory=list)
    reasons: list[str]
    positiveFactors: list[str] = Field(default_factory=list)
    negativeFactors: list[str] = Field(default_factory=list)
    suggestions: list[str]
    counterfactuals: list[dict[str, Any]] = Field(default_factory=list)
    factorBuckets: dict[str, float] = Field(default_factory=dict)
    mlProb: float
    cbesProb: float
    confidence: float
    riskScore: float
    explanation: str
    modelVersion: str


class PublicMetricsResponse(BaseModel):
    applicationsProcessed: int
    approvalSpeedup: float
    accuracy: float
    automationRate: int


class DashboardMetricsResponse(BaseModel):
    totalApplications: int
    approved: int
    rejected: int
    deferred: int
    averageProcessingTime: int
    approvalRate: int
    avgLoanAmount: int
    automationRate: int


class StatsResponse(BaseModel):
    totalApplications: int
    approved: int
    rejected: int
    deferred: int
    approvalRate: float
    rejectionRate: float
    deferralRate: float
    averageCBES: float
    averageMLProbability: float


class ModelMetricItem(BaseModel):
    model: str
    accuracy: float
    precision: float
    recall: float
    auc: float
    f1: float = 0.0
    rank: int = 0
    tuned: bool = True


class ModelPredictionSummaryItem(BaseModel):
    model: str
    approveCount: int
    rejectCount: int
    accuracyFromCases: float


class ModelCaseItem(BaseModel):
    applicantId: str
    yTrue: int
    expectedDecision: str
    hybridDecision: str
    hybridConfidence: float
    approvalThreshold: float
    rejectionThreshold: float
    cbesProb: float
    bestModelProb: float
    modelProbabilities: dict[str, float]
    modelPredictions: dict[str, str]


class ModelAnalysisSummary(BaseModel):
    totalCases: int
    deferredCases: int
    deferralRate: float
    automatedCoverage: float
    automatedAccuracy: float
    overallHybridAccuracy: float
    bestModel: str = ""
    selectedAlpha: float = 0.25


class ModelConfusionItem(BaseModel):
    model: str
    tp: int
    fp: int
    tn: int
    fn: int
    f1FromCases: float


class ProbabilityBandItem(BaseModel):
    band: str
    approve: int
    reject: int
    defer: int
    total: int


class ModelAnalysisResponse(BaseModel):
    models: list[ModelMetricItem]
    modelsByProbabilityColumns: list[str]
    summary: ModelAnalysisSummary
    modelPredictionSummary: list[ModelPredictionSummaryItem]
    confusionByModel: list[ModelConfusionItem] = Field(default_factory=list)
    probabilityBands: list[ProbabilityBandItem] = Field(default_factory=list)
    cases: list[ModelCaseItem]
