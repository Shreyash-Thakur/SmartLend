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
