import { useEffect } from 'react'
import { useApplicationStore } from '@/store/applicationStore'

interface UseApplicationDataOptions {
  applicationId?: string
  scope?: 'all' | 'customer' | 'org'
}

export function useApplicationData(options?: string | UseApplicationDataOptions) {
  const applicationId = typeof options === 'string' ? options : options?.applicationId
  const scope = typeof options === 'string' ? 'all' : options?.scope ?? 'all'
  const {
    applications,
    selectedApplication,
    isLoading,
    error,
    loadApplications,
    loadApplication,
    addApplication,
    uploadDocument,
    overrideDecision,
  } = useApplicationStore()

  useEffect(() => {
    void loadApplications(scope)
  }, [loadApplications, scope])

  useEffect(() => {
    if (applicationId) {
      void loadApplication(applicationId)
    }
  }, [applicationId, loadApplication])

  return {
    applications,
    application: selectedApplication,
    isLoading,
    error,
    addApplication,
    uploadDocument,
    overrideDecision,
  }
}
