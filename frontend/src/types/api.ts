import type {
  ApplicationDecision,
  ApplicationStatus,
  DecisionType,
  LoanApplication,
  LoanApplicationFormData,
} from '@/types/application'

export interface ApiResponse<T> {
  success: boolean
  data: T
  error?: {
    code: string
    message: string
    details?: Record<string, unknown>
    timestamp: string
  }
  timestamp: string
}

export interface PaginatedResponse<T> {
  items: T[]
  total: number
  page: number
  pageSize: number
  hasMore: boolean
}

export interface CreateApplicationRequest {
  applicationData: LoanApplicationFormData
}

export interface CreateApplicationResponse {
  id: string
  status: ApplicationStatus
  createdAt: string
}

export interface UpdateApplicationRequest {
  applicationData: Partial<LoanApplicationFormData>
}

export interface FileUploadResponse {
  fileName: string
  documentType: string
  uploadedAt: string
  extractedData?: Record<string, unknown>
  fileSize: number
}

export interface GetDecisionResponse {
  decision: ApplicationDecision
}

export interface ManualDecisionRequest {
  status: DecisionType
  notes: string

  /**
   * Structured reviewer feedback for the relearning capture layer. Every field
   * is optional and maps 1:1 onto a column the backend's `ManualDecisionRequest`
   * already accepts — do not invent new ones here, they would be silently
   * dropped by Pydantic. Omitting a field leaves the column NULL rather than
   * guessed. See docs/RELEARNING-LOOP.md.
   */
  reviewerId?: string
  /** Self-rated 1-5 (Madras et al. scale); the backend rejects anything else. */
  reviewerConfidence?: number
  /** Measured by the UI from when the case was opened — never typed by hand. */
  timeSpentSeconds?: number
  /** Codes from `@/lib/reasonCodes`. */
  reasonCodes?: string[]
}

/** One reason code as it comes back on a decision report (already expanded). */
export interface ReportReasonCode {
  code: string
  label: string
  direction: string
  description: string
}

/** The human half of a decision report — null until a reviewer rules. */
export interface DecisionReportHumanReview {
  reviewId: string
  reviewerId: string | null
  decision: string
  reviewedAt: string | null
  reasonCodes: ReportReasonCode[]
  freeText: string | null
  reviewerConfidence: number | null
  timeSpentSeconds: number | null
  agreedWithEngine: boolean | null
  overrideDirection: string | null
  explorationFlag: boolean
  outcomeCensored: boolean
  realizedOutcome: number | null
}

export interface DecisionReportEngine {
  decision: string
  decisionReason: string
  selectedModel: string
  engineVersion: string | null
  thresholdArtifactHash: string | null
  pMl: number | null
  pCbes: number | null
  pBlend: number | null
  /** "captured" (read off the deferral row) | "derived" | "unavailable". */
  pBlendSource: string
  disagreement: number | null
  confidence: number | null
  confidenceLabel: string | null
  riskScore: number | null
  thresholds: { approve: number | null; reject: number | null; base: number | null }
  cbesBreakdown: Record<string, number | null>
  cbesWeights: Record<string, number | null>
  topFactors: Array<Record<string, unknown>>
  explanation: string
  routedToHumanReview: boolean
  explorationFlag: boolean
}

/** `GET /api/applications/{id}/report` — the per-application audit record. */
export interface DecisionReport {
  applicationId: string
  generatedAt: string
  application: {
    applicantId: string
    applicantName: string
    email: string
    phone: string
    createdAt: string | null
    status: string
    loanAmount: number | null
    loanPurpose: string
    loanTenureMonths: number
    data: Record<string, unknown>
  }
  engine: DecisionReportEngine
  humanReview: DecisionReportHumanReview | null
  analystNotes: string | null
  manualDecisionApplied: boolean
}

export interface ManualDecisionResponse {
  application: LoanApplication
  decidedAt: string
  decidedBy: string
}

export interface DashboardMetrics {
  totalApplications: number
  approved: number
  rejected: number
  deferred: number
  averageProcessingTime: number
  approvalRate: number
  avgLoanAmount: number
  automationRate: number
}

export interface PublicMetrics {
  applicationsProcessed: number
  approvalSpeedup: number
  accuracy: number
  automationRate: number
}

export interface ChartDataPoint {
  label: string
  value: number
  percentage?: number
}

export interface TrendDataPoint {
  date: string
  count: number
  approved?: number
  rejected?: number
  deferred?: number
}

export interface StatsResponse {
  totalApplications: number
  approved: number
  rejected: number
  deferred: number
  approvalRate: number
  rejectionRate: number
  deferralRate: number
  averageCBES: number
  averageMLProbability: number
}

export interface ModelMetricItem {
  model: string
  accuracy: number
  precision: number
  recall: number
  auc: number
  f1: number
  rank: number
  tuned: boolean
}

export interface ModelPredictionSummaryItem {
  model: string
  approveCount: number
  rejectCount: number
  accuracyFromCases: number
}

export interface ModelCaseItem {
  applicantId: string
  yTrue: number
  expectedDecision: 'APPROVE' | 'REJECT'
  hybridDecision: 'APPROVE' | 'REJECT' | 'DEFER'
  hybridConfidence: number
  approvalThreshold: number
  rejectionThreshold: number
  cbesProb: number
  bestModelProb: number
  modelProbabilities: Record<string, number>
  modelPredictions: Record<string, 'APPROVE' | 'REJECT'>
}

export interface ModelAnalysisSummary {
  totalCases: number
  deferredCases: number
  deferralRate: number
  automatedCoverage: number
  automatedAccuracy: number
  overallHybridAccuracy: number
  bestModel: string
  selectedAlpha: number
}

export interface ModelConfusionItem {
  model: string
  tp: number
  fp: number
  tn: number
  fn: number
  f1FromCases: number
}

export interface ProbabilityBandItem {
  band: string
  approve: number
  reject: number
  defer: number
  total: number
}

export interface ModelAnalysisResponse {
  models: ModelMetricItem[]
  modelsByProbabilityColumns: string[]
  summary: ModelAnalysisSummary
  modelPredictionSummary: ModelPredictionSummaryItem[]
  confusionByModel: ModelConfusionItem[]
  probabilityBands: ProbabilityBandItem[]
  cases: ModelCaseItem[]
}

export interface HealthResponse {
  status: 'ok'
  model: string
  auc: number
  t_base: number
  tau_d: number
}

export interface PredictionFeatureImpact {
  feature: string
  impact: number
}

export interface PredictionResponse {
  decision: 'APPROVE' | 'REJECT' | 'DEFER'
  confidence: number
  confidence_label: 'HIGH' | 'MEDIUM' | 'LOW'
  risk_score: number
  p_ml: number
  p_cbes: number
  disagreement: number
  decision_reason: string
  shap_explanation: PredictionFeatureImpact[]
  cbes_breakdown: {
    credit: number
    capacity: number
    behaviour: number
    liquidity: number
    stability: number
  }
}

export interface DashboardModelComparisonItem {
  model: string
  auc: number
  f1: number
  accuracy: number
  recall: number
  std_auc: number
}

export interface DashboardMetricsResponseV2 {
  baseline: {
    model: string
    auc: number
    accuracy: number
    f1: number
    recall: number
  }
  hybrid: {
    auc: number
    deferral_rate: number
    coverage: number
    non_deferred_accuracy: number
    non_deferred_f1: number
    approve_precision: number
    approve_recall: number
    reject_precision: number
    reject_recall: number
    t_base: number
    tau_d: number
  }
  improvement: {
    auc_delta: number
    accuracy_delta: number
  }
}

export interface ApplicationHistoryItem {
  id: string
  timestamp: string
  applicantId?: string
  applicantName?: string
  decision: 'APPROVE' | 'REJECT' | 'DEFER'
  confidence: number
  risk_score: number
  reason: string
}
