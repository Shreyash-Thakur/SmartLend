import { create } from 'zustand'
import type { LoanApplication, LoanApplicationFormData } from '@/types/application'
import {
  createApplication,
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
  loadApplications: (scope?: 'all' | 'customer' | 'org') => Promise<void>
  loadApplication: (applicationId: string) => Promise<void>
  addApplication: (payload: LoanApplicationFormData) => Promise<LoanApplication>
  uploadDocument: (applicationId: string, file: File) => Promise<void>
  overrideDecision: (applicationId: string, status: 'approved' | 'rejected' | 'deferred', notes: string) => Promise<void>
}

export const useApplicationStore = create<ApplicationStore>((set) => ({
  applications: [],
  selectedApplication: null,
  isLoading: false,
  error: null,
  loadApplications: async (scope = 'all') => {
    set({ isLoading: true, error: null })
    try {
      const applications = await getApplications(scope)
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
      set({
        applications,
        selectedApplication:
          applications.find((application) => application.id === applicationId) ?? null,
        isLoading: false,
      })
    } catch (error) {
      set({ isLoading: false, error: error instanceof Error ? error.message : 'Failed to upload document' })
      throw error
    }
  },
  overrideDecision: async (applicationId, status, notes) => {
    set({ isLoading: true, error: null })
    try {
      await submitManualDecision(applicationId, { status, notes })
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
}))
