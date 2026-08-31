import type {
  DecisionReport,
  FileUploadResponse,
  ModelAnalysisResponse,
  ManualDecisionRequest,
  PublicMetrics,
  StatsResponse,
  TrendDataPoint,
} from '@/types/api'
import { REASON_CODES, type ReasonCode } from '@/lib/reasonCodes'
import type {
  CustomerProfile,
  CustomerSample,
  LoanApplication,
  LoanApplicationFormData,
  ShortLoanApplicationFormData,
} from '@/types/application'
import { apiClient } from '@/services/api.client'

export interface RegionMetric {
  applications: number
  approved: number
  rejected: number
  deferred: number
  approvalRate: number
  rejectionRate: number
  deferralRate: number
}

export interface RegionMetricsResponse {
  regions: Record<string, RegionMetric>
  totalApplications: number
  updatedAt: string
}

export interface LocationMetricsResponse {
  areas: Record<string, RegionMetric>
  states: Record<string, RegionMetric>
  cities: Record<string, RegionMetric>
  totalApplications: number
  updatedAt: string
}

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

export async function getApplications(
  scope: 'all' | 'customer' | 'org' = 'all',
  applicantId?: string,
): Promise<LoanApplication[]> {
  try {
    const response = await apiClient.get<LoanApplication[]>('/applications', {
      params: {
        scope,
        applicant_id: applicantId,
      },
    })
    return response.data.map(normalizeApplication)
  } catch (error) {
    throw extractApiError(error)
  }
}

export async function getRegionMetrics(): Promise<RegionMetricsResponse> {
  try {
    const response = await apiClient.get<RegionMetricsResponse>('/region-metrics')
    return response.data
  } catch (error) {
    throw extractApiError(error)
  }
}

export async function getLocationMetrics(): Promise<LocationMetricsResponse> {
  try {
    const response = await apiClient.get<LocationMetricsResponse>('/location-metrics')
    return response.data
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

/** Fetch the demographic + bureau block the bank already holds for a customer.
 *  Returns null for an unknown id (HTTP 404) so the caller can say so plainly
 *  instead of rendering a half-empty panel. */
export async function getCustomerProfile(customerId: string): Promise<CustomerProfile | null> {
  try {
    const response = await apiClient.get<CustomerProfile>(`/customers/${encodeURIComponent(customerId)}/profile`)
    return response.data
  } catch (error) {
    if (
      typeof error === 'object' && error !== null && 'response' in error
      && (error as { response?: { status?: number } }).response?.status === 404
    ) {
      return null
    }
    throw extractApiError(error)
  }
}

/** Fetch a handful of example customer ids to offer as one-click chips.
 *
 *  Graceful degradation: this endpoint is additive and may not be deployed yet.
 *  Any failure at all — 404, 500, network, or a payload we cannot understand —
 *  resolves to an empty array so the caller simply renders no chips. It must
 *  never surface an error, and must never gate the form. */
export async function getCustomerSamples(): Promise<CustomerSample[]> {
  try {
    const response = await apiClient.get<unknown>('/customers/samples')
    const payload = response.data
    // Tolerate both a bare array and the `{ samples: [...] }` envelope, since
    // the exact shape is owned by another service.
    const rows: unknown = Array.isArray(payload)
      ? payload
      : (payload as { samples?: unknown; customers?: unknown } | null)?.samples
        ?? (payload as { customers?: unknown } | null)?.customers
    if (!Array.isArray(rows)) return []

    return rows
      .map((row): CustomerSample | null => {
        if (typeof row === 'string' || typeof row === 'number') {
          return { customer_id: String(row), descriptor: '' }
        }
        if (typeof row !== 'object' || row === null) return null
        const record = row as Record<string, unknown>
        const id = record.customer_id ?? record.customerId ?? record.id ?? record.sk_id_curr
        if (id === undefined || id === null || String(id).trim() === '') return null
        const descriptor =
          record.descriptor ?? record.description ?? record.label ?? record.summary ?? record.note
        return {
          customer_id: String(id).trim(),
          descriptor: typeof descriptor === 'string' ? descriptor : '',
        }
      })
      .filter((sample): sample is CustomerSample => sample !== null)
      .slice(0, 12)
  } catch {
    return []
  }
}

export async function createApplication(
  formData: LoanApplicationFormData | ShortLoanApplicationFormData,
): Promise<LoanApplication> {
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

/** Fetch the per-application audit record (engine half + human half).
 *
 *  Returns null for an unknown application (HTTP 404) so the caller can say so
 *  plainly. A case that simply has no human review yet is NOT a 404 — the
 *  backend answers 200 with `humanReview: null`, which is the common state for
 *  a freshly deferred application. */
export async function getDecisionReport(applicationId: string): Promise<DecisionReport | null> {
  try {
    const response = await apiClient.get<DecisionReport>(`/applications/${applicationId}/report`)
    return response.data
  } catch (error) {
    if (
      typeof error === 'object' && error !== null && 'response' in error
      && (error as { response?: { status?: number } }).response?.status === 404
    ) {
      return null
    }
    throw extractApiError(error)
  }
}

/** The reviewer reason-code taxonomy, from the backend that will store the codes.
 *
 *  Any failure resolves to the bundled copy in `@/lib/reasonCodes` rather than
 *  an error: a reviewer must never be shown a decision form with no reason
 *  checkboxes, because a decision with no recorded reason is exactly what this
 *  whole flow exists to prevent. */
export async function getReasonCodeCatalog(): Promise<ReasonCode[]> {
  try {
    const response = await apiClient.get<{ reasonCodes?: unknown }>('/review-reason-codes')
    const rows = response.data?.reasonCodes
    if (!Array.isArray(rows) || rows.length === 0) return REASON_CODES
    const parsed = rows
      .map((row): ReasonCode | null => {
        if (typeof row !== 'object' || row === null) return null
        const record = row as Record<string, unknown>
        const code = typeof record.code === 'string' ? record.code : ''
        if (!code) return null
        return {
          code,
          label: typeof record.label === 'string' ? record.label : code,
          direction: (record.direction === 'approve' || record.direction === 'reject' || record.direction === 'either')
            ? record.direction
            : 'either',
          description: typeof record.description === 'string' ? record.description : '',
        }
      })
      .filter((entry): entry is ReasonCode => entry !== null)
    return parsed.length > 0 ? parsed : REASON_CODES
  } catch {
    return REASON_CODES
  }
}

export async function deleteApplicationDocument(
  applicationId: string,
  documentId: string,
): Promise<void> {
  try {
    await apiClient.delete(`/applications/${applicationId}/documents/${documentId}`)
  } catch (error) {
    throw extractApiError(error)
  }
}

export async function bulkDecision(
  applicationIds: string[],
  status: 'approved' | 'rejected',
  notes: string,
): Promise<void> {
  await Promise.all(
    applicationIds.map((id) =>
      submitManualDecision(id, { status, notes }),
    ),
  )
}

export async function getActiveModel(): Promise<string> {
  try {
    const response = await apiClient.get<{ active_model: string }>('/model-analysis/active')
    return response.data.active_model
  } catch (error) {
    console.error('Failed to get active model:', error)
    return 'LogisticRegression'
  }
}

export async function setActiveModel(modelName: string): Promise<string> {
  try {
    const response = await apiClient.post<{ active_model: string }>('/model-analysis/active', { model_name: modelName })
    return response.data.active_model
  } catch (error) {
    throw extractApiError(error)
  }
}
