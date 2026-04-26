import React, { useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { FileText, Trash2, Eye, ChevronLeft, CheckCircle2, XCircle, Clock } from 'lucide-react'
import { DashboardLayout } from '@/components/layouts/DashboardLayout'
import { Card, Button, Textarea } from '@/components/common'
import { DecisionBanner, FeatureContributionChart, DecisionExplanation } from '@/components/sections'
import { useApplicationData } from '@/hooks/useApplicationData'
import { formatCurrency } from '@/lib/utils'

import { useAuth } from '@/hooks/useAuth'

type AnalystDecision = 'approved' | 'rejected' | null

export const ApplicationReview: React.FC = () => {
  const { applicationId } = useParams()
  const navigate = useNavigate()
  const { role } = useAuth()
  const { application, overrideDecision, deleteDocument, isLoading, error } = useApplicationData(applicationId)
  const [manualDecision, setManualDecision] = useState<AnalystDecision>(null)
  const [notes, setNotes] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [submitError, setSubmitError] = useState<string | null>(null)
  const [previewDoc, setPreviewDoc] = useState<{ name: string; dataUrl: string; mimeType: string } | null>(null)
  const [deletingDocId, setDeletingDocId] = useState<string | null>(null)

  if (isLoading && !application) {
    return (
      <DashboardLayout title="Application Review" role="organization">
        <Card><div className="py-16 text-center text-neutral-500">Loading application…</div></Card>
      </DashboardLayout>
    )
  }

  if ((error && !application) || !application) {
    return (
      <DashboardLayout title="Application Review" role="organization">
        <Card>
          <div className="py-16 text-center">
            <p className="text-red-700 mb-4">{error ?? 'Application not found'}</p>
            <Button variant="secondary" onClick={() => navigate(-1)}>← Go Back</Button>
          </div>
        </Card>
      </DashboardLayout>
    )
  }

  const handleSubmitDecision = async () => {
    if (!manualDecision || !notes.trim()) {
      setSubmitError('Please select a decision and provide notes before submitting.')
      return
    }
    setSubmitError(null)
    setIsSubmitting(true)
    try {
      await overrideDecision(application.id, manualDecision, notes)
      navigate('/review')
    } catch (e) {
      setSubmitError(e instanceof Error ? e.message : 'Submission failed')
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleDeleteDocument = async (docId: string) => {
    setDeletingDocId(docId)
    try {
      await deleteDocument(application.id, docId)
    } finally {
      setDeletingDocId(null)
    }
  }

  const documents: Array<{ id: string; fileName: string; documentType: string; fileSize: number; uploadedAt: string; dataUrl?: string; mimeType?: string }> =
    Array.isArray(application.documents) ? application.documents as never : []

  const statusIsDeferred = application.modelRecommendation === 'deferred'
  const statusIsApproved = application.modelRecommendation === 'approved'
  const statusIsRejected = application.modelRecommendation === 'rejected'
  const orgConfirmed = Boolean(application.manualDecisionApplied)

  return (
    <DashboardLayout title={`Application Review: ${application.id.slice(0, 12)}`} role="organization">
      {/* Document Preview Modal */}
      {previewDoc && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4"
          onClick={() => setPreviewDoc(null)}
        >
          <div
            className="relative max-w-4xl w-full max-h-[90vh] bg-white rounded-2xl shadow-2xl overflow-hidden flex flex-col"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between px-6 py-4 border-b border-neutral-200">
              <p className="font-semibold text-neutral-900 truncate">{previewDoc.name}</p>
              <button
                onClick={() => setPreviewDoc(null)}
                className="ml-4 text-neutral-400 hover:text-neutral-700 text-2xl leading-none"
              >×</button>
            </div>
            <div className="flex-1 overflow-auto p-4">
              {previewDoc.mimeType?.startsWith('image/') ? (
                <img src={previewDoc.dataUrl} alt={previewDoc.name} className="max-w-full mx-auto rounded" />
              ) : previewDoc.mimeType === 'application/pdf' ? (
                <iframe
                  src={previewDoc.dataUrl}
                  title={previewDoc.name}
                  className="w-full h-[70vh] rounded"
                />
              ) : (
                <div className="flex flex-col items-center justify-center py-16 gap-4 text-neutral-500">
                  <FileText className="h-16 w-16 text-neutral-300" />
                  <p>Preview not available for this file type.</p>
                  <a
                    href={previewDoc.dataUrl}
                    download={previewDoc.name}
                    className="px-4 py-2 bg-primary-600 text-white rounded-lg text-sm font-medium hover:bg-primary-700"
                  >
                    Download File
                  </a>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      <div className="mb-6">
        <Button variant="ghost" leftIcon={<ChevronLeft className="h-4 w-4" />} onClick={() => navigate(role === 'customer' ? '/dashboard/customer' : '/review')}>
          {role === 'customer' ? 'Back to Dashboard' : 'Back to Review Queue'}
        </Button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left Panel */}
        <aside className="lg:col-span-1 space-y-6">
          <Card title="Applicant Information" level="light">
            <div className="space-y-3">
              <DetailRow label="Name" value={application.applicantName} />
              <DetailRow label="Email" value={application.email} />
              <DetailRow label="Phone" value={application.phone} />
              <DetailRow
                label="City / Region"
                value={`${application.applicationData.city ?? 'N/A'} / ${application.applicationData.region ?? 'N/A'}`}
              />
              <div className="border-t pt-3 space-y-3">
                <DetailRow label="Age" value={application.applicationData.age} />
                <DetailRow label="Monthly Income" value={formatCurrency(application.applicationData.monthlyIncome)} />
                <DetailRow label="EMI Obligations" value={formatCurrency(application.applicationData.emi)} />
                <DetailRow label="Total Assets" value={formatCurrency(application.applicationData.assets)} />
                <DetailRow
                  label="Credit Score"
                  value={application.applicationData.creditScore?.toString() ?? 'N/A'}
                />
              </div>
              <div className="border-t pt-3 space-y-3">
                <DetailRow label="Loan Amount" value={formatCurrency(application.loanAmount)} />
                <DetailRow label="Purpose" value={application.loanPurpose} />
                <DetailRow label="Tenure" value={`${application.loanTenure} months`} />
                <DetailRow
                  label="Debt / Income"
                  value={`${application.applicationData.debtToIncomeRatio?.toFixed(1) ?? '0'}%`}
                />
              </div>
            </div>
          </Card>

          {/* Documents Panel */}
          <Card title={`Supporting Documents (${documents.length})`} level="light">
            {documents.length === 0 ? (
              <p className="text-sm text-neutral-500 py-4 text-center">No documents attached</p>
            ) : (
              <ul className="space-y-3">
                {documents.map((doc) => (
                  <li key={doc.id} className="flex items-center gap-3 p-3 bg-neutral-50 rounded-xl border border-neutral-200">
                    <FileText className="h-5 w-5 text-blue-500 shrink-0" />
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-neutral-900 truncate">{doc.fileName}</p>
                      <p className="text-xs text-neutral-500 uppercase">{doc.documentType} · {Math.round(doc.fileSize / 1024)} KB</p>
                    </div>
                    <div className="flex items-center gap-1">
                      {doc.dataUrl && (
                        <button
                          onClick={() => setPreviewDoc({ name: doc.fileName, dataUrl: doc.dataUrl!, mimeType: doc.mimeType ?? '' })}
                          className="p-1.5 rounded-lg text-neutral-500 hover:text-blue-600 hover:bg-blue-50 transition-colors"
                          title="Preview"
                        >
                          <Eye className="h-4 w-4" />
                        </button>
                      )}
                      <button
                        onClick={() => void handleDeleteDocument(doc.id)}
                        disabled={deletingDocId === doc.id}
                        className="p-1.5 rounded-lg text-neutral-500 hover:text-red-600 hover:bg-red-50 transition-colors disabled:opacity-40"
                        title="Remove"
                      >
                        <Trash2 className="h-4 w-4" />
                      </button>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </Card>
        </aside>

        {/* Right Panel */}
        <section className="lg:col-span-2 space-y-6">
          {/* DOMINANT DECISION CARD (HEAVY NEOBRUTALISM) */}
          <div className={`rounded p-6 border-4 border-[#000000] shadow-[8px_8px_0px_#000000] ${
            statusIsApproved ? 'bg-[#B0F0DA]' :
            statusIsRejected ? 'bg-[#FF6B6B]' :
            'bg-[#FD9745]'
          }`}>
            <div className="flex flex-col md:flex-row md:items-start justify-between gap-4">
              <div className="flex items-start gap-4">
                {statusIsApproved ? <CheckCircle2 className="h-10 w-10 text-[#000000] shrink-0 mt-1" /> :
                 statusIsRejected ? <XCircle className="h-10 w-10 text-[#000000] shrink-0 mt-1" /> :
                 <Clock className="h-10 w-10 text-[#000000] shrink-0 mt-1" />}
                <div>
                  <h2 className="text-4xl font-black uppercase tracking-tight text-[#000000]">
                    {statusIsApproved ? 'Approve' :
                     statusIsRejected ? 'Reject' :
                     'Defer'}
                  </h2>
                  <p className="text-[#000000] font-bold text-lg mt-1">
                    CONFIDENCE: <span className="uppercase font-black bg-white px-2 py-0.5 rounded border-2 border-[#000000] shadow-[2px_2px_0px_#000000] ml-2">{application.decision?.confidence ?? 'N/A'}</span>
                  </p>
                  <p className="text-[#000000] font-medium mt-4 max-w-xl leading-relaxed">
                    {application.decision?.explanation ?? (statusIsApproved ? 'Application meets all criteria for automated approval.' : statusIsRejected ? 'Application falls below acceptable risk thresholds.' : 'Application requires human review due to borderline signals.')}
                  </p>
                  
                  {/* Top 3 Features */}
                  {application.decision?.featureImportance && application.decision.featureImportance.length > 0 && (
                    <div className="mt-5 flex flex-wrap gap-2 items-center">
                      <span className="text-sm font-black text-[#000000] mr-2">TOP FACTORS:</span>
                      {application.decision.featureImportance.slice(0, 3).map((f) => (
                        <span key={f.name} className="px-3 py-1 rounded border-2 border-[#000000] text-xs font-black bg-white text-[#000000] shadow-[2px_2px_0px_#000000]">
                          {f.name.replace(/_/g, ' ')} ({f.impact > 0 ? '+' : ''}{f.impact.toFixed(2)})
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              </div>
              
              <div className="text-right shrink-0 mt-4 md:mt-0 flex flex-col items-end gap-3">
                <div className="inline-block px-3 py-1.5 bg-white border-2 border-[#000000] rounded font-black text-sm text-[#000000] shadow-[4px_4px_0px_#000000]">
                  {orgConfirmed ? '✓ CONFIRMED' : 'AWAITING ANALYST'}
                </div>
                {application.ml_prob !== undefined && application.cbes_prob !== undefined && Math.abs(application.ml_prob - application.cbes_prob) > 0.15 && (
                  <div className="px-2 py-1 bg-[#FD9745] border-2 border-[#000000] shadow-[2px_2px_0px_#000000] rounded text-xs font-black text-[#000000]">
                    ⚠️ MODEL DISAGREEMENT
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* XAI / Decision breakdown */}
          {application.decision && (
            <>
              <Card title="Decision Analysis">
                <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
                  <ScoreTile label="Risk Score" value={`${(application.decision.riskScore * 100).toFixed(1)}%`} color="red" />
                  <ScoreTile label="CBES Score" value={`${application.decision.cbessScore}/100`} color="blue" />
                  <ScoreTile label="Confidence" value={application.decision.confidence ?? 'medium'} color="green" isText />
                </div>
              </Card>
              <FeatureContributionChart features={application.decision.featureImportance} />
              <DecisionExplanation decision={application.decision} />
              {role !== 'customer' && application.decision.allModelPredictions && Object.keys(application.decision.allModelPredictions).length > 0 && (
                <Card title="Ensemble Predictions Deep Dive" description="Approval probabilities from all underlying models (for research purposes).">
                  <div className="grid grid-cols-2 gap-4 md:grid-cols-3">
                    {Object.entries(application.decision.allModelPredictions).map(([model, prob]) => (
                      <div key={model} className={`rounded border-2 border-[#000000] p-4 flex flex-col justify-between shadow-[2px_2px_0px_#000000] ${prob >= 0.5 ? 'bg-[#B0F0DA]' : 'bg-[#FF6B6B]'} text-[#000000]`}>
                        <p className="text-xs font-black uppercase tracking-wider">{model}</p>
                        <div className="mt-2 text-xl font-black bg-white rounded px-2 py-0.5 border-2 border-[#000000] inline-block self-start">
                          {(prob * 100).toFixed(1)}%
                        </div>
                      </div>
                    ))}
                  </div>
                </Card>
              )}
            </>
          )}

          {/* Analyst Decision Section — always visible for org */}
          {!orgConfirmed && role !== 'customer' ? (
            <Card
              title={statusIsDeferred ? 'Analyst Decision Required' : 'Override or Confirm Model Decision'}
              className={`border-2 ${statusIsDeferred ? 'border-amber-300 bg-amber-50/40' : 'border-neutral-200'}`}
            >
              <div className="space-y-5">
                {statusIsDeferred && (
                  <div className="rounded-xl bg-amber-100 border border-amber-200 p-4 text-sm text-amber-800">
                    This case was deferred by the model — it requires a human decision before the customer can proceed.
                  </div>
                )}
                <div>
                  <label className="block text-sm font-semibold text-neutral-900 mb-3">
                    {statusIsDeferred ? 'Analyst Decision' : 'Override Decision (optional)'}
                  </label>
                  <div className="space-y-3">
                    <DecisionOption
                      label="✓ Approve"
                      description="Approve the loan application"
                      value="approved"
                      checked={manualDecision === 'approved'}
                      onChange={() => setManualDecision('approved')}
                      color="green"
                    />
                    <DecisionOption
                      label="✗ Reject"
                      description="Reject the loan application"
                      value="rejected"
                      checked={manualDecision === 'rejected'}
                      onChange={() => setManualDecision('rejected')}
                      color="red"
                    />
                  </div>
                </div>

                <Textarea
                  label="Decision Notes *"
                  placeholder="Provide your reasoning. This will be visible to the applicant on their dashboard."
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                  rows={4}
                  required
                />

                {submitError && (
                  <p className="text-sm text-red-600 bg-red-50 border border-red-200 rounded-lg px-3 py-2">{submitError}</p>
                )}

                <div className="flex gap-3">
                  <Button
                    variant="secondary"
                    onClick={() => navigate('/review')}
                    className="flex-1"
                  >
                    Cancel
                  </Button>
                  <Button
                    variant="primary"
                    className="flex-1"
                    isLoading={isSubmitting}
                    onClick={() => void handleSubmitDecision()}
                    disabled={!manualDecision || !notes.trim()}
                  >
                    {isSubmitting ? 'Submitting…' : 'Submit Decision'}
                  </Button>
                </div>
              </div>
            </Card>
          ) : (
            <Card title="Org Decision Recorded" className={`border-2 ${
              application.status === 'rejected' ? 'border-red-200 bg-red-50/40' :
              application.status === 'deferred' ? 'border-amber-200 bg-amber-50/40' :
              'border-green-200 bg-green-50/40'
            }`}>
              <div className="flex items-start gap-4">
                {application.status === 'rejected' ? (
                  <XCircle className="h-6 w-6 text-red-600 shrink-0 mt-0.5" />
                ) : application.status === 'deferred' ? (
                  <Clock className="h-6 w-6 text-amber-600 shrink-0 mt-0.5" />
                ) : (
                  <CheckCircle2 className="h-6 w-6 text-green-600 shrink-0 mt-0.5" />
                )}
                <div>
                  <p className={`font-semibold ${
                    application.status === 'rejected' ? 'text-red-900' :
                    application.status === 'deferred' ? 'text-amber-900' :
                    'text-green-900'
                  }`}>
                    Decision: {application.status.charAt(0).toUpperCase() + application.status.slice(1)}
                  </p>
                  {application.analystNotes && (
                    <p className={`mt-2 text-sm rounded-lg px-3 py-2 ${
                      application.status === 'rejected' ? 'text-red-800 bg-red-100' :
                      application.status === 'deferred' ? 'text-amber-800 bg-amber-100' :
                      'text-green-800 bg-green-100'
                    }`}>
                      "{application.analystNotes}"
                    </p>
                  )}
                </div>
              </div>
            </Card>
          )}
        </section>
      </div>
    </DashboardLayout>
  )
}

function DetailRow({ label, value }: { label: string; value: string | number }) {
  return (
    <div>
      <p className="text-xs text-neutral-500 mb-0.5">{label}</p>
      <p className="text-sm font-medium text-neutral-900">{value}</p>
    </div>
  )
}

function ScoreTile({ label, value, color, isText = false }: { label: string; value: string; color: string; isText?: boolean }) {
  const colorMap: Record<string, string> = {
    red: 'bg-[#FF6B6B] text-[#000000]',
    blue: 'bg-[#6E61FF] text-[#000000]',
    green: 'bg-[#B0F0DA] text-[#000000]',
  }
  return (
    <div className={`rounded border-2 border-[#000000] p-4 shadow-[2px_2px_0px_#000000] ${colorMap[color] ?? 'bg-white text-[#000000]'}`}>
      <p className="text-xs font-black uppercase tracking-wider mb-1 opacity-80">{label}</p>
      <p className={`font-black ${isText ? 'text-xl uppercase' : 'text-3xl'}`}>{value}</p>
    </div>
  )
}

function DecisionOption({
  label, description, value, checked, onChange, color,
}: {
  label: string; description: string; value: string; checked: boolean; onChange: () => void; color: 'green' | 'red'
}) {
  const ringColor = color === 'green' ? 'border-green-400 bg-green-50' : 'border-red-400 bg-red-50'
  return (
    <label className={`flex items-center gap-3 p-4 border-2 rounded-xl cursor-pointer transition-all ${
      checked ? ringColor : 'border-neutral-200 bg-white hover:border-neutral-300'
    }`}>
      <input
        type="radio"
        name="decision"
        value={value}
        checked={checked}
        onChange={onChange}
        className={`w-4 h-4 ${color === 'green' ? 'text-green-600' : 'text-red-600'}`}
      />
      <div>
        <p className="font-semibold text-neutral-900 text-sm">{label}</p>
        <p className="text-xs text-neutral-500 mt-0.5">{description}</p>
      </div>
    </label>
  )
}
