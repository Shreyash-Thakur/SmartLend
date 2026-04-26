import { useEffect } from 'react'
import { useApplicationStore } from '@/store/applicationStore'

interface UseApplicationDataOptions {
  applicationId?: string
  scope?: 'all' | 'customer' | 'org'
  applicantId?: string
}

export function useApplicationData(options?: string | UseApplicationDataOptions) {
  const applicationId = typeof options === 'string' ? options : options?.applicationId
  const scope = typeof options === 'string' ? 'all' : options?.scope ?? 'all'
  const applicantId = typeof options === 'string' ? undefined : options?.applicantId
  const {
    applications,
    selectedApplication,
    isLoading,
    error,
    loadApplications,
    loadApplication,
    addApplication,
    uploadDocument,
    deleteDocument,
    overrideDecision,
    bulkOverrideDecision,
  } = useApplicationStore()

  useEffect(() => {
    if (scope === 'customer' && !applicantId) {
      return
    }
    void loadApplications(scope, applicantId)
  }, [applicantId, loadApplications, scope])

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
    deleteDocument,
    overrideDecision,
    bulkOverrideDecision,
  }
}

