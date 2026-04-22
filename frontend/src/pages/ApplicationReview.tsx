import React, { useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { DashboardLayout } from '@/components/layouts/DashboardLayout'
import { Card, Button, Textarea } from '@/components/common'
import { DecisionBanner, FeatureContributionChart, DecisionExplanation } from '@/components/sections'
import { useApplicationData } from '@/hooks/useApplicationData'
import { useDecisionExplanation } from '@/hooks/useDecisionExplanation'
import { formatCurrency } from '@/lib/utils'

export const ApplicationReview: React.FC = () => {
  const { applicationId } = useParams()
  const navigate = useNavigate()
  const { application, overrideDecision, isLoading, error } = useApplicationData(applicationId)
  const [manualDecision, setManualDecision] = useState<'approved' | 'rejected' | null>(null)
  const [notes, setNotes] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)
  const decisionMeta = useDecisionExplanation(application?.decision)

  if (isLoading && !application) {
    return (
      <DashboardLayout title="Application Review">
        <Card>
          <div className="text-center py-12">
            <p className="text-neutral-600">Loading application...</p>
          </div>
        </Card>
      </DashboardLayout>
    )
  }

  if (error && !application) {
    return (
      <DashboardLayout title="Application Review">
        <Card>
          <div className="text-center py-12">
            <p className="text-red-700">{error}</p>
            <Button variant="secondary" onClick={() => navigate(-1)} className="mt-4">
              Go Back
            </Button>
          </div>
        </Card>
      </DashboardLayout>
    )
  }

  if (!application) {
    return (
      <DashboardLayout title="Application Review">
        <Card>
          <div className="text-center py-12">
            <p className="text-neutral-600">Application not found</p>
            <Button variant="secondary" onClick={() => navigate(-1)} className="mt-4">
              Go Back
            </Button>
          </div>
        </Card>
      </DashboardLayout>
    )
  }

  const handleSubmitDecision = async () => {
    if (!manualDecision || !notes.trim()) {
      return
    }

    setIsSubmitting(true)
    try {
      await overrideDecision(application.id, manualDecision, notes)
      navigate('/dashboard/org')
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <DashboardLayout title={`Application Review: ${application.id.slice(0, 8)}`}>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left: Applicant Details */}
        <aside className="lg:col-span-1">
          <Card title="Applicant Information">
            <div className="space-y-4">
              <DetailRow label="Name" value={application.applicantName} />
              <DetailRow label="Email" value={application.email} />
              <DetailRow label="Phone" value={application.phone} />
              <DetailRow label="City / Region" value={`${application.applicationData.city ?? 'N/A'} / ${application.applicationData.region ?? 'N/A'}`} />
              <div className="border-t border-neutral-200 pt-4">
                <DetailRow label="Age" value={application.applicationData.age} />
                <DetailRow label="Gender" value={application.applicationData.gender} />
                <DetailRow label="Education" value={application.applicationData.education ?? 'N/A'} />
                <DetailRow label="Monthly Income" value={formatCurrency(application.applicationData.monthlyIncome)} />
                <DetailRow label="Annual Income" value={formatCurrency(application.applicationData.annualIncome ?? 0)} />
                <DetailRow label="EMI Obligations" value={formatCurrency(application.applicationData.emi)} />
                <DetailRow label="Total Assets" value={formatCurrency(application.applicationData.assets)} />
                <DetailRow
                  label="Credit Score"
                  value={application.applicationData.creditScore ? application.applicationData.creditScore.toString() : 'N/A'}
                />
              </div>
              <div className="border-t border-neutral-200 pt-4">
                <DetailRow label="Requested Amount" value={formatCurrency(application.loanAmount)} />
                <DetailRow label="Loan Purpose" value={application.loanPurpose} />
                <DetailRow label="Tenure" value={`${application.loanTenure} months`} />
                <DetailRow label="Debt / Income Ratio" value={`${application.applicationData.debtToIncomeRatio?.toFixed(1) ?? '0'}%`} />
              </div>
            </div>
          </Card>
        </aside>

        {/* Right: Decision & XAI */}
        <section className="lg:col-span-2 space-y-6">
          {/* Decision Banner */}
          {application.decision && (
            <>
              <DecisionBanner
                status={application.decision.status}
                riskScore={application.decision.riskScore}
                confidence={application.decision.confidence}
                timestamp={application.decision.decidedAt}
                decidedBy={application.decision.decidedBy}
              />

              {/* Model Outputs */}
              <Card title="Model Outputs">
                <div className="grid grid-cols-1 gap-6 md:grid-cols-2 xl:grid-cols-4">
                  <div>
                    <p className="text-sm text-neutral-600 mb-2">Risk Score</p>
                    <div className="flex items-center gap-3">
                      <div className="flex-1 bg-neutral-200 rounded-full h-3 overflow-hidden">
                        <div
                          className="bg-red-500 h-full rounded-full transition-all"
                          style={{ width: `${application.decision.riskScore * 100}%` }}
                        />
                      </div>
                      <span className="text-lg font-bold text-neutral-900">{(application.decision.riskScore * 100).toFixed(1)}%</span>
                    </div>
                  </div>

                  <div>
                    <p className="text-sm text-neutral-600 mb-2">CBES Score</p>
                    <div className="flex items-center gap-3">
                      <div className="flex-1 bg-neutral-200 rounded-full h-3 overflow-hidden">
                        <div
                          className="bg-blue-500 h-full rounded-full transition-all"
                          style={{ width: `${application.decision.cbessScore}%` }}
                        />
                      </div>
                      <span className="text-lg font-bold text-neutral-900">{application.decision.cbessScore}/100</span>
                    </div>
                  </div>

                  <div>
                    <p className="text-sm text-neutral-600 mb-2">Uncertainty</p>
                    <div className="flex items-center gap-3">
                      <div className="flex-1 bg-neutral-200 rounded-full h-3 overflow-hidden">
                        <div
                          className="bg-amber-500 h-full rounded-full transition-all"
                          style={{ width: `${application.decision.uncertainty * 100}%` }}
                        />
                      </div>
                      <span className="text-lg font-bold text-neutral-900">{(application.decision.uncertainty * 100).toFixed(1)}%</span>
                    </div>
                  </div>

                  <div>
                    <p className="text-sm text-neutral-600 mb-2">Confidence Level</p>
                    <div className="rounded-xl bg-neutral-100 p-4">
                      <p className="text-lg font-bold text-neutral-900">{decisionMeta.confidenceLabel}</p>
                      <p className="text-sm text-neutral-600">{decisionMeta.confidencePercent}% effective confidence</p>
                    </div>
                  </div>
                </div>
              </Card>

              {/* Feature Importance */}
              <FeatureContributionChart features={application.decision.featureImportance} />

              {/* Decision Explanation */}
              <DecisionExplanation decision={application.decision} />
            </>
          )}

          {/* Manual Decision Override (if deferred) */}
          {application.status === 'deferred' && (
            <Card title="Analyst Decision" className="border-2 border-amber-200 bg-amber-50/50">
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-semibold text-neutral-900 mb-3">Override Decision</label>
                  <div className="space-y-3">
                    <label className="flex items-center gap-3 p-3 border border-neutral-300 rounded-lg cursor-pointer hover:bg-neutral-100">
                      <input
                        type="radio"
                        name="decision"
                        value="approved"
                        checked={manualDecision === 'approved'}
                        onChange={(e) => setManualDecision(e.target.value as 'approved')}
                        className="w-4 h-4 text-green-600"
                      />
                      <div>
                        <p className="font-medium text-neutral-900">✓ Approve</p>
                        <p className="text-xs text-neutral-600">Loan approved based on manual review</p>
                      </div>
                    </label>

                    <label className="flex items-center gap-3 p-3 border border-neutral-300 rounded-lg cursor-pointer hover:bg-neutral-100">
                      <input
                        type="radio"
                        name="decision"
                        value="rejected"
                        checked={manualDecision === 'rejected'}
                        onChange={(e) => setManualDecision(e.target.value as 'rejected')}
                        className="w-4 h-4 text-red-600"
                      />
                      <div>
                        <p className="font-medium text-neutral-900">✗ Reject</p>
                        <p className="text-xs text-neutral-600">Application rejected based on manual review</p>
                      </div>
                    </label>
                  </div>
                </div>

                <div>
                  <Textarea
                    label="Review Notes"
                    placeholder="Document your decision reasoning..."
                    value={notes}
                    onChange={(e) => setNotes(e.target.value)}
                    required
                    rows={4}
                  />
                </div>

                <Button
                  variant="primary"
                  fullWidth
                  isLoading={isSubmitting || isLoading}
                  onClick={handleSubmitDecision}
                >
                  {isSubmitting ? 'Submitting...' : 'Submit Decision'}
                </Button>
              </div>
            </Card>
          )}
        </section>
      </div>
    </DashboardLayout>
  )
}

interface DetailRowProps {
  label: string
  value: string | number
}

const DetailRow: React.FC<DetailRowProps> = ({ label, value }) => (
  <div>
    <p className="text-sm text-neutral-600 mb-1">{label}</p>
    <p className="font-medium text-neutral-900">{value}</p>
  </div>
)
