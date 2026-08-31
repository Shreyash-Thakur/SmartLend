import React, { useEffect, useMemo, useRef, useState } from 'react'
import { AlertTriangle, Clock, Info } from 'lucide-react'
import { Button, Card, Textarea } from '@/components/common'
import { getReasonCodeCatalog } from '@/services/applications'
import {
  CONFIDENCE_LABELS,
  OTHER_REASON_CODE,
  REASON_CODES,
  reasonCodesFor,
  type ReasonCode,
} from '@/lib/reasonCodes'
import type { ManualDecisionRequest } from '@/types/api'

type AnalystDecision = 'approved' | 'rejected'

export type ReviewerFeedback = Pick<
  ManualDecisionRequest,
  'reviewerId' | 'reviewerConfidence' | 'timeSpentSeconds' | 'reasonCodes'
>

interface ReviewerFeedbackFormProps {
  /** True when the engine deferred — changes the framing, not the requirements. */
  isDeferred: boolean
  /** Attribution for `deferred_reviews.reviewer_id`; falls back server-side. */
  reviewerId?: string
  isSubmitting?: boolean
  submitError?: string | null
  onCancel: () => void
  onSubmit: (decision: AnalystDecision, notes: string, feedback: ReviewerFeedback) => void | Promise<void>
}

function formatElapsed(totalSeconds: number): string {
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = Math.floor(totalSeconds % 60)
  return `${minutes}m ${String(seconds).padStart(2, '0')}s`
}

/**
 * The structured feedback a reviewer records when deciding a deferred case.
 *
 * WHY REASON CODES ARE MANDATORY. Submit stays disabled until at least one box
 * is ticked, and the form says so. A decision captured with no reason is a row
 * that can never contribute to the disparate-deferral check or to
 * reviewer-consistency analysis — collecting it is the entire purpose of the
 * capture layer, so a blank one is worse than useless: it inflates the counts
 * without adding evidence. See docs/RELEARNING-LOOP.md.
 *
 * WHY TIME-SPENT IS NOT A FIELD. It is measured from the moment this form
 * mounts (i.e. when the reviewer opened the case) to the moment they submit.
 * Asking a reviewer to self-report the number would produce a rounded guess,
 * and the reviewer-attention-degradation check needs a real measurement.
 *
 * All four values go out through fields `ManualDecisionRequest` already
 * accepts. Nothing here invents a new backend field.
 */
export const ReviewerFeedbackForm: React.FC<ReviewerFeedbackFormProps> = ({
  isDeferred,
  reviewerId,
  isSubmitting = false,
  submitError = null,
  onCancel,
  onSubmit,
}) => {
  const [decision, setDecision] = useState<AnalystDecision | null>(null)
  const [selectedCodes, setSelectedCodes] = useState<string[]>([])
  const [confidence, setConfidence] = useState<number>(3)
  const [notes, setNotes] = useState('')
  const [catalog, setCatalog] = useState<ReasonCode[]>(REASON_CODES)
  const [elapsedSeconds, setElapsedSeconds] = useState(0)

  // The clock starts when the reviewer opens the case, not when they start
  // typing. Kept in a ref so re-renders never restart it.
  const openedAt = useRef<number>(Date.now())

  useEffect(() => {
    const timer = window.setInterval(() => {
      setElapsedSeconds((Date.now() - openedAt.current) / 1000)
    }, 1000)
    return () => window.clearInterval(timer)
  }, [])

  useEffect(() => {
    let cancelled = false
    void getReasonCodeCatalog().then((entries) => {
      if (!cancelled) setCatalog(entries)
    })
    return () => {
      cancelled = true
    }
  }, [])

  const availableCodes = useMemo(
    () => (decision ? reasonCodesFor(decision, catalog) : []),
    [catalog, decision],
  )

  // Switching verdict drops any ticked code that no longer applies, so an
  // approve-only reason can never be filed against a rejection.
  const chooseDecision = (next: AnalystDecision) => {
    setDecision(next)
    const allowed = new Set(reasonCodesFor(next, catalog).map((entry) => entry.code))
    setSelectedCodes((current) => current.filter((code) => allowed.has(code)))
  }

  const toggleCode = (code: string) => {
    setSelectedCodes((current) =>
      current.includes(code) ? current.filter((entry) => entry !== code) : [...current, code],
    )
  }

  const otherSelected = selectedCodes.includes(OTHER_REASON_CODE)
  const missingReason = decision !== null && selectedCodes.length === 0
  const missingOtherText = otherSelected && !notes.trim()
  const canSubmit = decision !== null && selectedCodes.length > 0 && !missingOtherText && !isSubmitting

  const handleSubmit = () => {
    if (!decision || selectedCodes.length === 0 || missingOtherText) return
    // `notes` is doing double duty: it is stored as `human_free_text` AND shown
    // to the applicant as the decision explanation. When the reviewer relied on
    // the checkboxes alone, send the selected reasons in prose rather than an
    // empty string — otherwise the applicant's dashboard shows a decision with
    // no stated reason, which is the customer-facing version of the same
    // problem the reason codes exist to solve.
    const labelFor = (code: string) => catalog.find((entry) => entry.code === code)?.label ?? code
    const outgoingNotes = notes.trim() || `Reasons: ${selectedCodes.map(labelFor).join('; ')}.`
    void onSubmit(decision, outgoingNotes, {
      reasonCodes: selectedCodes,
      reviewerConfidence: confidence,
      timeSpentSeconds: Math.max(0, Math.round((Date.now() - openedAt.current) / 1000)),
      ...(reviewerId ? { reviewerId } : {}),
    })
  }

  return (
    <Card
      title={isDeferred ? 'Analyst Decision Required' : 'Override or Confirm Model Decision'}
      description="Your verdict, the reasons behind it, and how confident you are are all recorded against this case."
      className={`border-2 ${isDeferred ? 'border-amber-300 bg-amber-50/40' : 'border-neutral-200'}`}
    >
      <div className="space-y-6">
        {isDeferred && (
          <div className="rounded-xl bg-amber-100 border border-amber-200 p-4 text-sm text-amber-800">
            This case was deferred by the model — it requires a human decision before the customer can proceed.
          </div>
        )}

        {/* --- 1. verdict ------------------------------------------------- */}
        <div>
          <label className="block text-sm font-semibold text-neutral-900 mb-3">
            {isDeferred ? 'Analyst Decision' : 'Override Decision'}
          </label>
          <div className="space-y-3">
            <DecisionOption
              label="✓ Approve"
              description="Approve the loan application"
              value="approved"
              checked={decision === 'approved'}
              onChange={() => chooseDecision('approved')}
              color="green"
            />
            <DecisionOption
              label="✗ Reject"
              description="Reject the loan application"
              value="rejected"
              checked={decision === 'rejected'}
              onChange={() => chooseDecision('rejected')}
              color="red"
            />
          </div>
        </div>

        {/* --- 2. reason codes -------------------------------------------- */}
        <div>
          <div className="flex items-start justify-between gap-4 mb-2">
            <label className="block text-sm font-semibold text-neutral-900">
              Reasons for this decision <span className="text-red-500">*</span>
            </label>
            <span className="text-xs text-neutral-500 shrink-0">
              {selectedCodes.length} selected
            </span>
          </div>
          <p className="text-xs text-neutral-500 mb-3 flex items-start gap-1.5">
            <Info className="h-3.5 w-3.5 shrink-0 mt-0.5" />
            <span>
              Select at least one. A decision recorded without a reason cannot be audited or
              analysed later, so it is not accepted.
            </span>
          </p>

          {!decision ? (
            <div className="rounded-xl border border-dashed border-neutral-300 bg-neutral-50 px-4 py-6 text-center text-sm text-neutral-500">
              Choose Approve or Reject above to see the relevant reasons.
            </div>
          ) : (
            <div className="grid gap-2 sm:grid-cols-2">
              {availableCodes.map((entry) => {
                const checked = selectedCodes.includes(entry.code)
                return (
                  <label
                    key={entry.code}
                    title={entry.description}
                    className={`flex items-start gap-3 rounded-xl border-2 p-3 cursor-pointer transition-all ${
                      checked
                        ? 'border-primary-400 bg-primary-50'
                        : 'border-neutral-200 bg-white hover:border-neutral-300'
                    }`}
                  >
                    <input
                      type="checkbox"
                      checked={checked}
                      onChange={() => toggleCode(entry.code)}
                      className="mt-0.5 h-4 w-4 shrink-0 accent-primary-600"
                    />
                    <span className="min-w-0">
                      <span className="block text-sm font-medium text-neutral-900">{entry.label}</span>
                      <span className="mt-0.5 block text-xs leading-4 text-neutral-500">{entry.description}</span>
                    </span>
                  </label>
                )
              })}
            </div>
          )}

          {missingReason && (
            <p className="mt-3 flex items-center gap-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-800">
              <AlertTriangle className="h-4 w-4 shrink-0" />
              At least one reason is required before this decision can be submitted.
            </p>
          )}
        </div>

        {/* --- 3. confidence ---------------------------------------------- */}
        <div>
          <label className="block text-sm font-semibold text-neutral-900 mb-1">
            How confident are you in this decision?
          </label>
          <p className="text-xs text-neutral-500 mb-3">
            1 = very unsure, 5 = very confident. Used to model reviewer consistency; it never
            changes the outcome for this applicant.
          </p>
          <div className="flex flex-wrap gap-2">
            {[1, 2, 3, 4, 5].map((level) => (
              <button
                key={level}
                type="button"
                onClick={() => setConfidence(level)}
                aria-pressed={confidence === level}
                className={`flex-1 min-w-[92px] rounded-xl border-2 px-3 py-2 text-center transition-all ${
                  confidence === level
                    ? 'border-primary-400 bg-primary-50'
                    : 'border-neutral-200 bg-white hover:border-neutral-300'
                }`}
              >
                <span className="block text-lg font-semibold text-neutral-900">{level}</span>
                <span className="block text-[11px] leading-3 text-neutral-500">{CONFIDENCE_LABELS[level]}</span>
              </button>
            ))}
          </div>
        </div>

        {/* --- 4. free text ----------------------------------------------- */}
        <Textarea
          label={otherSelected ? 'Decision Notes' : 'Decision Notes (optional)'}
          required={otherSelected}
          placeholder={
            otherSelected
              ? 'You selected "Other" — describe the grounds for this decision.'
              : 'Anything the reason codes above do not capture. Visible to the applicant on their dashboard.'
          }
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          rows={4}
          error={missingOtherText ? 'Notes are required when "Other" is selected.' : undefined}
        />

        {/* --- 5. measured time ------------------------------------------- */}
        <div className="flex items-center gap-2 rounded-xl border border-neutral-200 bg-neutral-50 px-4 py-3 text-sm text-neutral-600">
          <Clock className="h-4 w-4 shrink-0 text-neutral-400" />
          <span>
            Time on this case: <span className="font-semibold text-neutral-900">{formatElapsed(elapsedSeconds)}</span>
            <span className="text-neutral-500"> — measured automatically and submitted with your decision.</span>
          </span>
        </div>

        {submitError && (
          <p className="text-sm text-red-600 bg-red-50 border border-red-200 rounded-lg px-3 py-2">{submitError}</p>
        )}

        <div className="flex gap-3">
          <Button variant="secondary" onClick={onCancel} className="flex-1">
            Cancel
          </Button>
          <Button
            variant="primary"
            className="flex-1"
            isLoading={isSubmitting}
            onClick={handleSubmit}
            disabled={!canSubmit}
          >
            {isSubmitting ? 'Submitting…' : 'Submit Decision'}
          </Button>
        </div>
      </div>
    </Card>
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
        className={`w-4 h-4 ${color === 'green' ? 'accent-green-600' : 'accent-red-600'}`}
      />
      <div>
        <p className="font-semibold text-neutral-900 text-sm">{label}</p>
        <p className="text-xs text-neutral-500 mt-0.5">{description}</p>
      </div>
    </label>
  )
}
