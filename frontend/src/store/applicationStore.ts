import { create } from 'zustand'
import type {
  LoanApplication,
  LoanApplicationFormData,
  ShortLoanApplicationFormData,
} from '@/types/application'
import type { ManualDecisionRequest } from '@/types/api'
import {
  bulkDecision,
  createApplication,
  deleteApplicationDocument,
  getApplicationById,
  getApplications,
  submitManualDecision,
  uploadApplicationDocument,
} from '@/services/applications'

interface ApplicationStore {
  applications: LoanApplication[]
  selectedApplication: LoanApplication | null
  isLoading: boolean
  error: string | null
  loadApplications: (scope?: 'all' | 'customer' | 'org', applicantId?: string) => Promise<void>
  loadApplication: (applicationId: string) => Promise<void>
  // Accepts either vocabulary: the redesigned short form, or the legacy
  // full form still used by document-parsed and seeded submissions.
  addApplication: (payload: LoanApplicationFormData | ShortLoanApplicationFormData) => Promise<LoanApplication>
  uploadDocument: (applicationId: string, file: File) => Promise<void>
  deleteDocument: (applicationId: string, documentId: string) => Promise<void>
  overrideDecision: (
    applicationId: string,
    status: 'approved' | 'rejected' | 'deferred',
    notes: string,
    feedback?: ReviewerFeedback,
  ) => Promise<void>
  bulkOverrideDecision: (applicationIds: string[], status: 'approved' | 'rejected', notes: string) => Promise<void>
}

/** Structured reviewer feedback for the relearning capture layer.
 *
 * Optional so the bulk-triage and legacy call sites keep working; when it is
 * absent the corresponding `deferred_reviews` columns are left NULL rather than
 * guessed. The keys are exactly the ones `ManualDecisionRequest` accepts. */
export type ReviewerFeedback = Pick<
  ManualDecisionRequest,
  'reviewerId' | 'reviewerConfidence' | 'timeSpentSeconds' | 'reasonCodes'
>

export const useApplicationStore = create<ApplicationStore>((set) => ({
  applications: [],
  selectedApplication: null,
  isLoading: false,
  error: null,
  loadApplications: async (scope = 'all', applicantId) => {
    set({ isLoading: true, error: null })
    try {
      const applications = await getApplications(scope, applicantId)
      set({ applications, isLoading: false })
    } catch (error) {
      set({ isLoading: false, error: error instanceof Error ? error.message : 'Failed to load applications' })
      throw error
    }
  },
  loadApplication: async (applicationId) => {
    set({ isLoading: true, error: null })
    try {
      const selectedApplication = await getApplicationById(applicationId)
      set({ selectedApplication, isLoading: false })
    } catch (error) {
      set({ isLoading: false, error: error instanceof Error ? error.message : 'Failed to load application' })
      throw error
    }
  },
  addApplication: async (payload) => {
    set({ isLoading: true, error: null })
    try {
      const application = await createApplication(payload)
      set((state) => ({
        applications: [application, ...state.applications],
        isLoading: false,
      }))
      return application
    } catch (error) {
      set({ isLoading: false, error: error instanceof Error ? error.message : 'Failed to create application' })
      throw error
    }
  },
  uploadDocument: async (applicationId, file) => {
    set({ isLoading: true, error: null })
    try {
      await uploadApplicationDocument(applicationId, file)
      const applications = await getApplications()
      const selectedApplication = await getApplicationById(applicationId)
      set({ applications, selectedApplication, isLoading: false })
    } catch (error) {
      set({ isLoading: false, error: error instanceof Error ? error.message : 'Failed to upload document' })
      throw error
    }
  },
  deleteDocument: async (applicationId, documentId) => {
    set({ isLoading: true, error: null })
    try {
      await deleteApplicationDocument(applicationId, documentId)
      const selectedApplication = await getApplicationById(applicationId)
      set((state) => ({
        selectedApplication,
        applications: state.applications.map((a) =>
          a.id === applicationId ? { ...a, documents: selectedApplication?.documents ?? [] } : a,
        ),
        isLoading: false,
      }))
    } catch (error) {
      set({ isLoading: false, error: error instanceof Error ? error.message : 'Failed to delete document' })
      throw error
    }
  },
  overrideDecision: async (applicationId, status, notes, feedback) => {
    set({ isLoading: true, error: null })
    try {
      await submitManualDecision(applicationId, { status, notes, ...(feedback ?? {}) })
      const applications = await getApplications()
      set({
        applications,
        selectedApplication:
          applications.find((application) => application.id === applicationId) ?? null,
        isLoading: false,
      })
    } catch (error) {
      set({ isLoading: false, error: error instanceof Error ? error.message : 'Failed to update decision' })
      throw error
    }
  },
  bulkOverrideDecision: async (applicationIds, status, notes) => {
    set({ isLoading: true, error: null })
    try {
      await bulkDecision(applicationIds, status, notes)
      const applications = await getApplications()
      set({ applications, isLoading: false })
    } catch (error) {
      set({ isLoading: false, error: error instanceof Error ? error.message : 'Failed to bulk update decisions' })
      throw error
    }
  },
}))

