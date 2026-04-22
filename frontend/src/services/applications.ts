import type {
  DashboardMetrics,
  FileUploadResponse,
  ManualDecisionRequest,
  PublicMetrics,
  TrendDataPoint,
} from '@/types/api'
import type { LoanApplication, LoanApplicationFormData } from '@/types/application'
import { dashboardMetrics, mockApplications, publicMetrics, trendData } from '@/lib/mockData'
import { apiClient } from '@/services/api.client'
import { simulateFileExtraction } from '@/services/fileParser'

let applicationState = [...mockApplications]

function wait<T>(value: T, timeout = 350): Promise<T> {
  return new Promise((resolve) => {
    window.setTimeout(() => resolve(value), timeout)
  })
}

async function withApiFallback<T>(apiCall: () => Promise<T>, fallback: () => Promise<T>): Promise<T> {
  try {
    return await apiCall()
  } catch {
    return fallback()
  }
}

function buildDashboardMetrics(applications: LoanApplication[]): DashboardMetrics {
  const approved = applications.filter((application) => application.status === 'approved').length
  const rejected = applications.filter((application) => application.status === 'rejected').length
  const deferred = applications.filter((application) => application.status === 'deferred').length
  const autoDecided = applications.filter((application) => application.decision?.decidedBy === 'model').length
  const avgLoanAmount =
    applications.length > 0
      ? Math.round(applications.reduce((sum, application) => sum + application.loanAmount, 0) / applications.length)
      : 0

  return {
    totalApplications: applications.length,
    approved,
    rejected,
    deferred,
    averageProcessingTime: dashboardMetrics.averageProcessingTime,
    approvalRate: applications.length > 0 ? Math.round((approved / applications.length) * 100) : 0,
    avgLoanAmount,
    automationRate: applications.length > 0 ? Math.round((autoDecided / applications.length) * 100) : 0,
  }
}

export async function getPublicMetrics(): Promise<PublicMetrics> {
  return withApiFallback(
    async () => {
      const response = await apiClient.get<PublicMetrics>('/public-metrics')
      return response.data
    },
    () => wait(publicMetrics),
  )
}

export async function getDashboardMetrics(): Promise<DashboardMetrics> {
  return withApiFallback(
    async () => {
      const response = await apiClient.get<DashboardMetrics>('/dashboard-metrics')
      return response.data
    },
    () => wait(buildDashboardMetrics(applicationState)),
  )
}

export async function getApplications(scope: 'all' | 'customer' | 'org' = 'all'): Promise<LoanApplication[]> {
  return withApiFallback(
    async () => {
      const response = await apiClient.get<LoanApplication[]>('/applications', {
        params: { scope },
      })
      applicationState = [...response.data]
      return response.data
    },
    async () => {
      if (scope === 'customer') {
        return wait(applicationState.filter((application) => application.source === 'customer'))
      }

      if (scope === 'org') {
        return wait([...applicationState])
      }

      return wait([...applicationState])
    },
  )
}

export async function getApplicationById(applicationId: string) {
  return withApiFallback(
    async () => {
      const response = await apiClient.get<LoanApplication>(`/applications/${applicationId}`)
      return response.data
    },
    () => wait(applicationState.find((application) => application.id === applicationId) ?? null),
  )
}

export async function getTrendData(): Promise<TrendDataPoint[]> {
  return withApiFallback(
    async () => {
      const response = await apiClient.get<TrendDataPoint[]>('/trends')
      return response.data
    },
    () => wait(trendData),
  )
}

export async function createApplication(
  formData: LoanApplicationFormData,
): Promise<LoanApplication> {
  return withApiFallback(
    async () => {
      const response = await apiClient.post<LoanApplication>('/applications', formData)
      applicationState = [response.data, ...applicationState]
      return response.data
    },
    async () => {
      const timestamp = new Date().toISOString()
      const applicantName = `${formData.firstName} ${formData.lastName}`.trim()
      const annualIncome = formData.annualIncome ?? formData.monthlyIncome * 12
      const totalAssets =
        formData.totalAssets ??
        (formData.residentialAssetsValue ?? 0) +
          (formData.commercialAssetsValue ?? 0) +
          (formData.bankBalance ?? 0)
      const totalEmi = (formData.emi ?? 0) + (formData.existingEmis ?? 0)
      const emiIncomeRatio =
        formData.emiIncomeRatio ?? (formData.monthlyIncome ? (totalEmi / formData.monthlyIncome) * 100 : 0)
      const loanIncomeRatio =
        formData.loanIncomeRatio ?? (annualIncome ? (formData.loanAmount / annualIncome) * 100 : 0)
      const debtToIncomeRatio =
        formData.debtToIncomeRatio ??
        (formData.monthlyIncome
          ? (((formData.liabilities ?? 0) + totalEmi * 12) / (formData.monthlyIncome * 12)) * 100
          : 0)
      const created: LoanApplication = {
        id: `app-${crypto.randomUUID()}`,
        createdAt: timestamp,
        updatedAt: timestamp,
        status: 'submitted',
        source: 'customer',
        applicantId: formData.applicantId ?? `cust-${crypto.randomUUID().slice(0, 8)}`,
        applicantName,
        email: formData.email,
        phone: formData.phone,
        loanAmount: formData.loanAmount,
        loanPurpose: formData.loanPurpose,
        loanTenure: formData.loanTenure,
        interestRate: formData.interestRate,
        applicationData: {
          firstName: formData.firstName,
          lastName: formData.lastName,
          gender: formData.gender,
          maritalStatus: formData.maritalStatus,
          education: formData.education,
          monthlyIncome: formData.monthlyIncome,
          annualIncome,
          emi: formData.emi,
          existingEmis: formData.existingEmis,
          employmentType: formData.employmentType,
          yearsOfEmployment: formData.yearsOfEmployment,
          assets: totalAssets,
          residentialAssetsValue: formData.residentialAssetsValue,
          commercialAssetsValue: formData.commercialAssetsValue,
          bankBalance: formData.bankBalance,
          totalAssets,
          liabilities: formData.liabilities,
          creditScore: formData.creditScore,
          creditHistory: formData.creditHistory,
          totalLoans: formData.totalLoans,
          activeLoans: formData.activeLoans,
          closedLoans: formData.closedLoans,
          missedPayments: formData.missedPayments,
          creditUtilizationRatio: formData.creditUtilizationRatio,
          emiIncomeRatio,
          loanIncomeRatio,
          debtToIncomeRatio,
          age: formData.age,
          dependents: formData.dependents,
          residenceType: formData.residenceType,
          region: formData.region,
          city: formData.city,
        },
        documents: [],
      }

      applicationState = [created, ...applicationState]
      return wait(created, 500)
    },
  )
}

export async function uploadApplicationDocument(
  applicationId: string,
  file: File,
): Promise<FileUploadResponse> {
  return withApiFallback(
    async () => {
      const formData = new FormData()
      formData.append('file', file)
      const response = await apiClient.post<FileUploadResponse>(
        `/applications/${applicationId}/documents`,
        formData,
        { headers: { 'Content-Type': 'multipart/form-data' } },
      )
      return response.data
    },
    async () => {
      const uploaded = await simulateFileExtraction(file)
      applicationState = applicationState.map((application) =>
        application.id === applicationId
          ? {
              ...application,
              updatedAt: new Date().toISOString(),
              documents: [
                ...(application.documents ?? []),
                {
                  id: crypto.randomUUID(),
                  fileName: uploaded.fileName,
                  documentType: uploaded.documentType as 'pdf' | 'csv' | 'jpg' | 'png',
                  fileSize: uploaded.fileSize,
                  uploadedAt: uploaded.uploadedAt,
                  extractedData: uploaded.extractedData,
                },
              ],
            }
          : application,
      )

      return wait(uploaded)
    },
  )
}

export async function submitManualDecision(
  applicationId: string,
  payload: ManualDecisionRequest,
) {
  return withApiFallback(
    async () => {
      const response = await apiClient.post<LoanApplication>(`/applications/${applicationId}/decision`, payload)
      applicationState = applicationState.map((entry) =>
        entry.id === applicationId ? response.data : entry,
      )
      return response.data
    },
    async () => {
      const application = applicationState.find((entry) => entry.id === applicationId)

      if (!application) {
        return wait(null)
      }

      const updated: LoanApplication = {
        ...application,
        status: payload.status,
        updatedAt: new Date().toISOString(),
        decision: {
          ...(application.decision ?? {
            id: crypto.randomUUID(),
            riskScore: 0.4,
            cbessScore: 70,
            uncertainty: 0.2,
            confidence: 'medium',
            explanation: '',
            positiveFactors: [],
            negativeFactors: [],
            featureImportance: [],
            modelVersion: 'cbes-v2',
          }),
          status: payload.status,
          decidedAt: new Date().toISOString(),
          decidedBy: 'human',
          explanation: payload.notes,
          analystNotes: payload.notes,
          analystId: 'analyst-placeholder',
        },
      }

      applicationState = applicationState.map((entry) =>
        entry.id === applicationId ? updated : entry,
      )

      return wait(updated, 450)
    },
  )
}
