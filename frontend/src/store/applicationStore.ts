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
  loadApplications: (scope?: 'all' | 'customer' | 'org') => Promise<void>
  loadApplication: (applicationId: string) => Promise<void>
  addApplication: (payload: LoanApplicationFormData) => Promise<LoanApplication>
  uploadDocument: (applicationId: string, file: File) => Promise<void>
  overrideDecision: (applicationId: string, status: 'approved' | 'rejected', notes: string) => Promise<void>
}

export const useApplicationStore = create<ApplicationStore>((set) => ({
  applications: [],
  selectedApplication: null,
  isLoading: false,
  loadApplications: async (scope = 'all') => {
    set({ isLoading: true })
    const applications = await getApplications(scope)
    set({ applications, isLoading: false })
  },
  loadApplication: async (applicationId) => {
    set({ isLoading: true })
    const selectedApplication = await getApplicationById(applicationId)
    set({ selectedApplication, isLoading: false })
  },
  addApplication: async (payload) => {
    set({ isLoading: true })
    const application = await createApplication(payload)
    set((state) => ({
      applications: [application, ...state.applications],
      isLoading: false,
    }))
    return application
  },
  uploadDocument: async (applicationId, file) => {
    set({ isLoading: true })
    await uploadApplicationDocument(applicationId, file)
    const applications = await getApplications()
    set({
      applications,
      selectedApplication:
        applications.find((application) => application.id === applicationId) ?? null,
      isLoading: false,
    })
  },
  overrideDecision: async (applicationId, status, notes) => {
    set({ isLoading: true })
    await submitManualDecision(applicationId, { status, notes })
    const applications = await getApplications()
    set({
      applications,
      selectedApplication:
        applications.find((application) => application.id === applicationId) ?? null,
      isLoading: false,
    })
  },
}))
