import React, { useMemo, useState } from 'react'
import { CheckCircle2, Clock3, FileText, XCircle } from 'lucide-react'
import { DashboardLayout } from '@/components/layouts/DashboardLayout'
import { Badge, Button, Card } from '@/components/common'
import { useApplicationData } from '@/hooks/useApplicationData'
import type { DecisionType, LoanApplication } from '@/types/application'
import { formatCurrency, formatDate } from '@/lib/utils'

export const ReviewPage: React.FC = () => {
  const { applications, overrideDecision, isLoading, error } = useApplicationData({ scope: 'org' })
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [actionLoading, setActionLoading] = useState(false)

  const deferredApplications = useMemo(
    () => applications.filter((application) => application.status === 'deferred' && application.source === 'customer'),
    [applications],
  )

  const selectedApplication = useMemo(
    () => deferredApplications.find((application) => application.id === selectedId) ?? deferredApplications[0] ?? null,
    [deferredApplications, selectedId],
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

  const bulkDecide = async (status: DecisionType) => {
    setActionLoading(true)
    try {
      for (const application of deferredApplications) {
        await overrideDecision(application.id, status, `Bulk analyst action: ${status}.`)
      }
      setSelectedId(null)
    } finally {
      setActionLoading(false)
    }
  }

  return (
    <DashboardLayout title="Review Queue" role="organization">
      <section className="mb-8 rounded-[32px] border border-[#d6e7e4] bg-white p-8 shadow-sm">
        <div className="flex flex-col gap-5 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <p className="text-xs uppercase tracking-[0.28em] text-neutral-500">Analyst Review</p>
            <h2 className="mt-3 text-4xl font-semibold tracking-tight text-neutral-900">Deferred applications</h2>
            <p className="mt-3 max-w-2xl text-neutral-600">
              Review customer applications that the hybrid ML and CBES engine deferred for human decisioning.
            </p>
          </div>
          <div className="flex flex-wrap gap-3">
            <Button
              variant="primary"
              leftIcon={<CheckCircle2 className="h-4 w-4" />}
              disabled={!deferredApplications.length || actionLoading}
              onClick={() => void bulkDecide('approved')}
            >
              Accept All
            </Button>
            <Button
              variant="secondary"
              leftIcon={<XCircle className="h-4 w-4" />}
              disabled={!deferredApplications.length || actionLoading}
              onClick={() => void bulkDecide('rejected')}
            >
              Reject All
            </Button>
          </div>
        </div>
      </section>

      {error && (
        <section className="mb-8">
          <Card className="border-red-200 bg-red-50">
            <p className="text-red-700">{error}</p>
          </Card>
        </section>
      )}

      <section className="grid gap-6 lg:grid-cols-[0.9fr_1.1fr]">
        <Card title={`Review Queue (${deferredApplications.length})`}>
          {isLoading ? (
            <p className="text-neutral-600">Loading applications...</p>
          ) : deferredApplications.length === 0 ? (
            <div className="py-12 text-center">
              <Clock3 className="mx-auto h-10 w-10 text-neutral-400" />
              <p className="mt-3 font-medium text-neutral-900">No deferred customer applications</p>
              <p className="mt-1 text-sm text-neutral-500">New deferred submissions will appear here.</p>
            </div>
          ) : (
            <div className="space-y-3">
              {deferredApplications.map((application) => (
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
                    <Badge status="deferred" />
                  </div>
                  <div className="mt-3 grid grid-cols-3 gap-2 text-xs text-neutral-600">
                    <span>ML {(application.ml_prob ?? 0).toFixed(3)}</span>
                    <span>CBES {(application.cbes_score ?? application.cbes_prob ?? 0).toFixed(3)}</span>
                    <span>Conf {(application.confidence ?? 0).toFixed(3)}</span>
                  </div>
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
              </div>

              <div className="flex flex-wrap gap-3">
                <Button
                  variant="primary"
                  leftIcon={<CheckCircle2 className="h-4 w-4" />}
                  disabled={actionLoading}
                  onClick={() => void decide(selectedApplication, 'approved')}
                >
                  Accept
                </Button>
                <Button
                  variant="secondary"
                  leftIcon={<XCircle className="h-4 w-4" />}
                  disabled={actionLoading}
                  onClick={() => void decide(selectedApplication, 'rejected')}
                >
                  Reject
                </Button>
                <Button
                  variant="ghost"
                  leftIcon={<Clock3 className="h-4 w-4" />}
                  disabled={actionLoading}
                  onClick={() => void decide(selectedApplication, 'deferred')}
                >
                  Defer
                </Button>
              </div>
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
