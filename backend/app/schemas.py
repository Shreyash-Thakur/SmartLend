from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ManualDecisionRequest(BaseModel):
    status: str
    notes: str = ""


class LoanApplicationInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    firstName: str = "Customer"
    lastName: str = "Applicant"
    email: str = "customer@example.com"
    phone: str = "+91 9000000000"
    loanAmount: float
    loanPurpose: str = "personal"
    loanTenure: int = 36
    interestRate: float = 12.0
    monthlyIncome: float
    annualIncome: float | None = None
    emi: float = 0.0
    existingEmis: float = 0.0
    residentialAssetsValue: float = 0.0
    commercialAssetsValue: float = 0.0
    bankBalance: float = 0.0
    cibilScore: int = 650
    totalLoans: int = 0
    activeLoans: int = 0
    closedLoans: int = 0
    missedPayments: int = 0
    creditUtilizationRatio: float = 0.0
    age: int
    dependents: int = 0
    employmentType: str = "salaried"
    yearsOfEmployment: int = 0
    region: str = "west"
    city: str = "Unknown"
    gender: str = "other"
    maritalStatus: str = "single"

    @field_validator("age")
    @classmethod
    def validate_age(cls, value: int) -> int:
        if value < 18 or value > 70:
            raise ValueError("age must be between 18 and 70")
        return value

    @field_validator("monthlyIncome")
    @classmethod
    def validate_income(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("monthlyIncome must be greater than 0")
        return value

    @field_validator("loanAmount")
    @classmethod
    def validate_loan_amount(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("loanAmount must be greater than 0")
        return value

    @field_validator("cibilScore")
    @classmethod
    def validate_cibil(cls, value: int) -> int:
        if value < 300 or value > 900:
            raise ValueError("cibilScore must be between 300 and 900")
        return value

    @model_validator(mode="after")
    def validate_emi_to_income(self) -> "LoanApplicationInput":
        if self.emi > self.monthlyIncome:
            raise ValueError("emi must be less than or equal to monthlyIncome")
        if self.annualIncome is None:
            self.annualIncome = float(self.monthlyIncome) * 12
        return self


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
