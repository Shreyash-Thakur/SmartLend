import React, { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { CheckCircle2, Clock3, FileText, XCircle } from 'lucide-react'
import { DashboardLayout } from '@/components/layouts/DashboardLayout'
import { Badge, Button, Card } from '@/components/common'
import { useApplicationData } from '@/hooks/useApplicationData'
import type { DecisionType, LoanApplication } from '@/types/application'
import { formatCurrency, formatDate } from '@/lib/utils'

export const ReviewPage: React.FC = () => {
  const navigate = useNavigate()
  const { applications, overrideDecision, isLoading, error } = useApplicationData({ scope: 'org' })
  const [activeStatus, setActiveStatus] = useState<'all' | 'submitted' | 'approved' | 'rejected' | 'deferred'>('all')
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [actionLoading, setActionLoading] = useState(false)

  const reviewApplications = useMemo(() => applications, [applications])

  const statusCounts = useMemo(
    () => ({
      submitted: reviewApplications.filter((application) => application.status === 'submitted').length,
      approved: reviewApplications.filter((application) => application.status === 'approved').length,
      rejected: reviewApplications.filter((application) => application.status === 'rejected').length,
      deferred: reviewApplications.filter((application) => application.status === 'deferred').length,
    }),
    [reviewApplications],
  )

  const visibleApplications = useMemo(() => {
    if (activeStatus === 'all') {
      return reviewApplications
    }
    return reviewApplications.filter((application) => application.status === activeStatus)
  }, [activeStatus, reviewApplications])

  const manualQueue = useMemo(
    () => reviewApplications.filter(
      (application) =>
        application.source === 'customer'
        && (application.status === 'submitted' || application.status === 'deferred'),
    ),
    [reviewApplications],
  )

  const recommendedApproved = useMemo(
    () => manualQueue.filter((application) => application.modelRecommendation === 'approved'),
    [manualQueue],
  )

  const recommendedRejected = useMemo(
    () => manualQueue.filter((application) => application.modelRecommendation === 'rejected'),
    [manualQueue],
  )

  const selectedApplication = useMemo(
    () => visibleApplications.find((application) => application.id === selectedId) ?? visibleApplications[0] ?? null,
    [visibleApplications, selectedId],
  )

  const decide = async (application: LoanApplication, status: DecisionType) => {
    setActionLoading(true)
    try {
      await overrideDecision(application.id, status, `Analyst marked application as ${status}.`)
      if (selectedId === application.id) {
        setSelectedId(null)
      }
    } finally {
      setActionLoading(false)
    }
  }

  const bulkDecide = async (items: LoanApplication[], status: DecisionType) => {
    setActionLoading(true)
    try {
      for (const application of items) {
        await overrideDecision(application.id, status, `Bulk analyst action: ${status}.`)
      }
      setSelectedId(null)
    } finally {
      setActionLoading(false)
    }
  }

  const approveAllPending = async () => {
    await bulkDecide(manualQueue, 'approved')
  }

  const rejectAllPending = async () => {
    await bulkDecide(manualQueue, 'rejected')
  }

  return (
    <DashboardLayout title="Review Queue" role="organization">
      <section className="mb-8 rounded-[32px] border border-[#d6e7e4] bg-white p-8 shadow-sm">
        <div className="flex flex-col gap-5 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <p className="text-xs uppercase tracking-[0.28em] text-neutral-500">Analyst Review</p>
            <h2 className="mt-3 text-4xl font-semibold tracking-tight text-neutral-900">End-to-end decision desk</h2>
            <p className="mt-3 max-w-2xl text-neutral-600">
              Review submitted and deferred applications, inspect model evidence and uploaded files, and apply bulk or manual decisions.
            </p>
          </div>
          <div className="flex flex-wrap gap-3">
            <Button variant="ghost" onClick={() => navigate('/dashboard/models')}>
              Open Model Analysis
            </Button>
            <Button
              variant="primary"
              leftIcon={<CheckCircle2 className="h-4 w-4" />}
              disabled={!recommendedApproved.length || actionLoading}
              onClick={() => void bulkDecide(recommendedApproved, 'approved')}
            >
              Accept All Recommended Approvals
            </Button>
            <Button
              variant="secondary"
              leftIcon={<XCircle className="h-4 w-4" />}
              disabled={!recommendedRejected.length || actionLoading}
              onClick={() => void bulkDecide(recommendedRejected, 'rejected')}
            >
              Reject All Recommended Rejections
            </Button>
            <Button
              variant="secondary"
              disabled={!manualQueue.length || actionLoading}
              onClick={() => void approveAllPending()}
            >
              Approve All Pending
            </Button>
            <Button
              variant="ghost"
              disabled={!manualQueue.length || actionLoading}
              onClick={() => void rejectAllPending()}
            >
              Reject All Pending
            </Button>
          </div>
        </div>
      </section>

      <section className="mb-6 grid gap-3 md:grid-cols-5">
        {[
          { key: 'all', label: 'All', count: reviewApplications.length },
          { key: 'submitted', label: 'Submitted', count: statusCounts.submitted },
          { key: 'approved', label: 'Approved', count: statusCounts.approved },
          { key: 'rejected', label: 'Rejected', count: statusCounts.rejected },
          { key: 'deferred', label: 'Deferred', count: statusCounts.deferred },
        ].map((item) => (
          <button
            key={item.key}
            type="button"
            onClick={() => setActiveStatus(item.key as typeof activeStatus)}
            className={`rounded-xl border px-4 py-3 text-left transition-colors ${
              activeStatus === item.key
                ? 'border-primary-300 bg-primary-50'
                : 'border-neutral-200 bg-white hover:bg-neutral-50'
            }`}
          >
            <p className="text-xs uppercase tracking-[0.16em] text-neutral-500">{item.label}</p>
            <p className="mt-1 text-xl font-semibold text-neutral-900">{item.count}</p>
          </button>
        ))}
      </section>

      {error && (
        <section className="mb-8">
          <Card className="border-red-200 bg-red-50">
            <p className="text-red-700">{error}</p>
          </Card>
        </section>
      )}

      <section className="grid gap-6 lg:grid-cols-[0.9fr_1.1fr]">
        <Card title={`Application Queue (${visibleApplications.length})`}>
          {isLoading ? (
            <p className="text-neutral-600">Loading applications...</p>
          ) : visibleApplications.length === 0 ? (
            <div className="py-12 text-center">
              <Clock3 className="mx-auto h-10 w-10 text-neutral-400" />
              <p className="mt-3 font-medium text-neutral-900">No applications for this status</p>
              <p className="mt-1 text-sm text-neutral-500">Switch status filters to inspect approved/rejected/deferred records.</p>
            </div>
          ) : (
            <div className="space-y-3">
              {visibleApplications.map((application) => (
                <button
                  key={application.id}
                  type="button"
                  onClick={() => setSelectedId(application.id)}
                  className={`w-full rounded-2xl border p-4 text-left transition-colors ${
                    selectedApplication?.id === application.id
                      ? 'border-primary-300 bg-primary-50'
                      : 'border-neutral-200 bg-white hover:bg-neutral-50'
                  }`}
                >
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <p className="font-semibold text-neutral-900">{application.applicantName}</p>
                      <p className="mt-1 text-sm text-neutral-500">{formatCurrency(application.loanAmount)}</p>
                    </div>
                    <Badge status={application.status === 'submitted' ? 'pending' : application.status} />
                  </div>
                  <div className="mt-3 grid grid-cols-3 gap-2 text-xs text-neutral-600">
                    <span>ML {(application.ml_prob ?? 0).toFixed(3)}</span>
                    <span>CBES {(application.cbes_score ?? application.cbes_prob ?? 0).toFixed(3)}</span>
                    <span>Conf {(application.confidence ?? 0).toFixed(3)}</span>
                  </div>
                  <p className="mt-2 text-xs text-neutral-500">
                    Model recommendation: {application.modelRecommendation ?? 'submitted'}
                  </p>
                </button>
              ))}
            </div>
          )}
        </Card>

        <Card title="Application Details">
          {!selectedApplication ? (
            <div className="py-12 text-center text-neutral-500">
              <FileText className="mx-auto h-10 w-10" />
              <p className="mt-3">Select a deferred application to review.</p>
            </div>
          ) : (
            <div className="space-y-6">
              <div className="grid gap-4 md:grid-cols-2">
                <Detail label="Applicant" value={selectedApplication.applicantName} />
                <Detail label="Submitted" value={formatDate(selectedApplication.createdAt)} />
                <Detail label="Loan Amount" value={formatCurrency(selectedApplication.loanAmount)} />
                <Detail label="Loan Purpose" value={selectedApplication.loanPurpose} />
                <Detail label="Monthly Income" value={formatCurrency(selectedApplication.applicationData.monthlyIncome)} />
                <Detail label="CIBIL Score" value={String(selectedApplication.applicationData.cibilScore ?? selectedApplication.applicationData.creditScore ?? 'N/A')} />
              </div>

              <div className="grid gap-4 md:grid-cols-3">
                <Metric label="ML Probability" value={(selectedApplication.ml_prob ?? 0).toFixed(4)} />
                <Metric label="CBES Score" value={(selectedApplication.cbes_score ?? selectedApplication.cbes_prob ?? 0).toFixed(4)} />
                <Metric label="Confidence" value={(selectedApplication.confidence ?? 0).toFixed(4)} />
              </div>

              <div className="rounded-2xl border border-neutral-200 bg-neutral-50 p-4">
                <p className="text-sm font-semibold text-neutral-900">Model Explanation</p>
                <p className="mt-2 text-sm leading-6 text-neutral-600">
                  {selectedApplication.decision?.explanation ?? 'Hybrid ML and CBES engine deferred this application for analyst review.'}
                </p>
                <p className="mt-2 text-xs text-neutral-500">
                  Recommended action: {selectedApplication.modelRecommendation ?? 'submitted'}
                </p>
              </div>

              <div className="rounded-2xl border border-neutral-200 bg-white p-4">
                <p className="text-sm font-semibold text-neutral-900">Uploaded Files</p>
                {selectedApplication.documents && selectedApplication.documents.length > 0 ? (
                  <div className="mt-3 space-y-2">
                    {selectedApplication.documents.map((document) => (
                      <div key={document.id} className="rounded-lg border border-neutral-200 bg-neutral-50 px-3 py-2 text-sm">
                        <p className="font-medium text-neutral-900">{document.fileName}</p>
                        <p className="text-xs text-neutral-600">Type: {document.documentType} | Uploaded: {formatDate(document.uploadedAt)}</p>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="mt-2 text-sm text-neutral-600">No documents were uploaded for this application.</p>
                )}
              </div>

              <div className="flex flex-wrap gap-3">
                <Button
                  variant="primary"
                  leftIcon={<CheckCircle2 className="h-4 w-4" />}
                  disabled={actionLoading || selectedApplication.status === 'approved' || selectedApplication.source !== 'customer'}
                  onClick={() => void decide(selectedApplication, 'approved')}
                >
                  Accept
                </Button>
                <Button
                  variant="secondary"
                  leftIcon={<XCircle className="h-4 w-4" />}
                  disabled={actionLoading || selectedApplication.status === 'rejected' || selectedApplication.source !== 'customer'}
                  onClick={() => void decide(selectedApplication, 'rejected')}
                >
                  Reject
                </Button>
                <Button
                  variant="ghost"
                  leftIcon={<Clock3 className="h-4 w-4" />}
                  disabled={actionLoading || selectedApplication.status === 'deferred' || selectedApplication.source !== 'customer'}
                  onClick={() => void decide(selectedApplication, 'deferred')}
                >
                  Defer
                </Button>
              </div>

              {selectedApplication.source !== 'customer' && (
                <p className="text-xs text-neutral-500">
                  This is a training-data record. It is visible for analysis but cannot be manually overridden.
                </p>
              )}
            </div>
          )}
        </Card>
      </section>
    </DashboardLayout>
  )
}

function Detail({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl bg-neutral-50 p-4">
      <p className="text-xs uppercase tracking-[0.18em] text-neutral-500">{label}</p>
      <p className="mt-2 font-semibold capitalize text-neutral-900">{value}</p>
    </div>
  )
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl border border-neutral-200 bg-white p-4 shadow-sm">
      <p className="text-sm text-neutral-500">{label}</p>
      <p className="mt-2 text-2xl font-semibold text-neutral-900">{value}</p>
    </div>
  )
}
