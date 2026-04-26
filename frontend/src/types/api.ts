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
