import type {
  FileUploadResponse,
  ModelAnalysisResponse,
  ManualDecisionRequest,
  PublicMetrics,
  StatsResponse,
  TrendDataPoint,
} from '@/types/api'
import type { LoanApplication, LoanApplicationFormData } from '@/types/application'
import { apiClient } from '@/services/api.client'

function asNumber(value: unknown, fallback = 0): number {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value
  }
  if (typeof value === 'string') {
    const parsed = Number(value)
    return Number.isFinite(parsed) ? parsed : fallback
  }
  return fallback
}

function mapFinalDecisionToStatus(finalDecision?: string): LoanApplication['status'] {
  const decision = (finalDecision ?? '').toUpperCase()
  if (decision === 'APPROVE') return 'approved'
  if (decision === 'REJECT') return 'rejected'
  if (decision === 'DEFER') return 'deferred'
  return 'submitted'
}

function mapFinalDecisionToRecommendation(finalDecision?: string): LoanApplication['modelRecommendation'] {
  const decision = (finalDecision ?? '').toUpperCase()
  if (decision === 'APPROVE') return 'approved'
  if (decision === 'REJECT') return 'rejected'
  if (decision === 'DEFER') return 'deferred'
  return 'submitted'
}

function extractApiError(error: unknown): Error {
  if (
    typeof error === 'object' &&
    error !== null &&
    'code' in error &&
    (error as { code?: string }).code === 'ECONNABORTED'
  ) {
    return new Error('Request timed out. Please check that the backend is running on port 8000.')
  }
  if (
    typeof error === 'object' &&
    error !== null &&
    'message' in error &&
    typeof (error as { message?: string }).message === 'string' &&
    (error as { message: string }).message.toLowerCase().includes('network error')
  ) {
    return new Error('Unable to connect to backend service. Start the API server and retry.')
  }
  if (
    typeof error === 'object' &&
    error !== null &&
    'response' in error &&
    typeof (error as { response?: unknown }).response === 'object'
  ) {
    const response = (error as { response?: { data?: { error?: string; details?: string } } }).response
    const details = response?.data?.details
    const code = response?.data?.error
    if (details || code) {
      return new Error([code, details].filter(Boolean).join(': '))
    }
  }
  if (error instanceof Error) {
    return error
  }
  return new Error('Request failed')
}

function normalizeApplication(application: LoanApplication): LoanApplication {
  const applicationData = (application.applicationData ?? {}) as Record<string, unknown>
  const residentialAssetsValue = asNumber(applicationData.residentialAssetsValue)
  const commercialAssetsValue = asNumber(applicationData.commercialAssetsValue)
  const bankBalance = asNumber(applicationData.bankBalance)
  const assets =
    asNumber(applicationData.assets) ||
    asNumber(applicationData.totalAssets) ||
    residentialAssetsValue + commercialAssetsValue + bankBalance

  const decisionMeta = (applicationData._decision_meta ?? {}) as Record<string, unknown>
  const engineeredFeatures = (decisionMeta.engineered_features ?? {}) as Record<string, unknown>
  const debtToIncomeRatio =
    asNumber(applicationData.debtToIncomeRatio) ||
    asNumber(applicationData.debt_to_income_ratio) ||
    asNumber(engineeredFeatures.debt_to_income_ratio) * 100

  const creditScore = asNumber(applicationData.creditScore) || asNumber(applicationData.cibilScore)
  const rootConfidence = asNumber((application as unknown as Record<string, unknown>).confidence)

  return {
    ...application,
    source: application.source ?? 'customer',
    status: application.status ?? mapFinalDecisionToStatus(application.finalDecision),
    modelRecommendation:
      ((application as unknown as Record<string, unknown>).modelRecommendation as LoanApplication['modelRecommendation'])
      ?? mapFinalDecisionToRecommendation(application.finalDecision),
    manualDecisionApplied: Boolean((application as unknown as Record<string, unknown>).manualDecisionApplied),
    ml_prob: asNumber(application.ml_prob),
    cbes_prob: asNumber(application.cbes_prob),
    cbes_score: asNumber(application.cbes_score) || asNumber(application.cbes_prob),
    confidence: rootConfidence,
    applicationData: {
      ...applicationData,
      assets,
      totalAssets: asNumber(applicationData.totalAssets) || assets,
      creditScore: creditScore || undefined,
      debtToIncomeRatio: debtToIncomeRatio || undefined,
      monthlyIncome: asNumber(applicationData.monthlyIncome),
      emi: asNumber(applicationData.emi),
      age: asNumber(applicationData.age, 0),
      firstName: String(applicationData.firstName ?? application.applicantName ?? 'Applicant'),
      lastName: String(applicationData.lastName ?? ''),
      gender: String(applicationData.gender ?? 'other').toLowerCase() as LoanApplication['applicationData']['gender'],
      employmentType: String(applicationData.employmentType ?? 'salaried').toLowerCase() as LoanApplication['applicationData']['employmentType'],
    },
  }
}

export async function getPublicMetrics(): Promise<PublicMetrics> {
  try {
    const response = await apiClient.get<PublicMetrics>('/public-metrics')
    return response.data
  } catch (error) {
    throw extractApiError(error)
  }
}

export async function getStats(): Promise<StatsResponse> {
  try {
    const response = await apiClient.get<StatsResponse>('/stats')
    return response.data
  } catch (error) {
    throw extractApiError(error)
  }
}

export async function getModelAnalysis(limit = 300): Promise<ModelAnalysisResponse> {
  try {
    const response = await apiClient.get<ModelAnalysisResponse>('/model-analysis', { params: { limit } })
    return response.data
  } catch (error) {
    throw extractApiError(error)
  }
}

export async function getApplications(scope: 'all' | 'customer' | 'org' = 'all'): Promise<LoanApplication[]> {
  try {
    const response = await apiClient.get<LoanApplication[]>('/applications', { params: { scope } })
    return response.data.map(normalizeApplication)
  } catch (error) {
    throw extractApiError(error)
  }
}

export async function getApplicationById(applicationId: string): Promise<LoanApplication | null> {
  try {
    const response = await apiClient.get<LoanApplication>(`/applications/${applicationId}`)
    return normalizeApplication(response.data)
  } catch (error) {
    throw extractApiError(error)
  }
}

export async function getTrendData(): Promise<TrendDataPoint[]> {
  try {
    const response = await apiClient.get<TrendDataPoint[]>('/trends')
    return response.data
  } catch (error) {
    throw extractApiError(error)
  }
}

export async function createApplication(formData: LoanApplicationFormData): Promise<LoanApplication> {
  try {
    const response = await apiClient.post<LoanApplication>('/applications', formData)
    return normalizeApplication(response.data)
  } catch (error) {
    throw extractApiError(error)
  }
}

export async function uploadApplicationDocument(
  applicationId: string,
  file: File,
): Promise<FileUploadResponse> {
  try {
    const formData = new FormData()
    formData.append('file', file)
    const response = await apiClient.post<FileUploadResponse>(
      `/applications/${applicationId}/documents`,
      formData,
      { headers: { 'Content-Type': 'multipart/form-data' } },
    )
    return response.data
  } catch (error) {
    throw extractApiError(error)
  }
}

export async function submitManualDecision(
  applicationId: string,
  payload: ManualDecisionRequest,
): Promise<LoanApplication | null> {
  try {
    const response = await apiClient.post<LoanApplication>(`/applications/${applicationId}/decision`, payload)
    return normalizeApplication(response.data)
  } catch (error) {
    throw extractApiError(error)
  }
}
