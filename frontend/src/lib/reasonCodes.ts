/**
 * Reviewer reason-code taxonomy — the boxes an analyst ticks when deciding a
 * deferred case.
 *
 * This list mirrors `backend/app/services/review_reason_codes.py`, which is the
 * source of truth. The review screen fetches the live catalogue from
 * `GET /api/review-reason-codes` and only falls back to this copy if that call
 * fails, so a backend outage degrades to possibly-stale labels rather than a
 * review screen with no checkboxes at all — and the codes actually written to
 * `deferred_reviews.human_reason_codes` stay identical either way.
 *
 * `direction` decides which boxes are offered for which verdict: an APPROVE
 * shows the approve-supporting and neutral codes, a REJECT shows the
 * reject-supporting and neutral ones. That filtering is a usability aid, not
 * validation — the backend deliberately accepts any code string.
 */

export type ReasonDirection = 'approve' | 'reject' | 'either' | 'unknown'

export interface ReasonCode {
  code: string
  label: string
  direction: ReasonDirection
  description: string
}

/** Ticking this code is what reveals the free-text box as required. */
export const OTHER_REASON_CODE = 'GEN-OTHER'

export const REASON_CODES: ReasonCode[] = [
  // --- supporting APPROVE ---
  { code: 'APR-EMP-STABLE', label: 'Stable employment', direction: 'approve', description: 'Continuous employment / business vintage supports servicing ability.' },
  { code: 'APR-REPAY-STRONG', label: 'Strong repayment history', direction: 'approve', description: 'Existing and closed credit lines were serviced without delinquency.' },
  { code: 'APR-COLLATERAL', label: 'Adequate collateral', direction: 'approve', description: 'Security offered covers the exposure at an acceptable margin.' },
  { code: 'APR-INCOME-HEADROOM', label: 'Sufficient income headroom', direction: 'approve', description: 'Residual income after the proposed EMI leaves comfortable margin.' },
  { code: 'APR-RELATIONSHIP', label: 'Long-standing customer', direction: 'approve', description: 'Established relationship with observed conduct on prior facilities.' },
  { code: 'APR-GUARANTOR', label: 'Guarantor strength', direction: 'approve', description: "Guarantor's standing materially reduces the residual risk." },
  // --- supporting REJECT ---
  { code: 'REJ-OBLIGATIONS', label: 'High existing obligations', direction: 'reject', description: 'Current debt service already absorbs too much of income.' },
  { code: 'REJ-THIN-FILE', label: 'Thin or no credit file', direction: 'reject', description: 'Too little bureau history to evidence repayment behaviour.' },
  { code: 'REJ-DELINQUENCY', label: 'Recent delinquency', direction: 'reject', description: 'Overdue or written-off accounts within the recent window.' },
  { code: 'REJ-EMI-BURDEN', label: 'Income insufficient for EMI', direction: 'reject', description: 'Proposed instalment exceeds demonstrable repayment capacity.' },
  { code: 'REJ-DOCS', label: 'Unverifiable documents', direction: 'reject', description: 'Submitted proofs are missing, inconsistent, or could not be verified.' },
  { code: 'REJ-UTILISATION', label: 'High credit utilisation', direction: 'reject', description: 'Revolving lines are drawn close to their sanctioned limits.' },
  // --- direction-neutral ---
  { code: 'GEN-MODEL-MISMATCH', label: 'Model score inconsistent with the file', direction: 'either', description: "The engine's score does not match what the documents show." },
  { code: 'GEN-POLICY-EXCEPTION', label: 'Policy exception applied', direction: 'either', description: 'Decision rests on an approved deviation from standard policy.' },
  { code: OTHER_REASON_CODE, label: 'Other', direction: 'either', description: 'Grounds not covered above — describe them in the notes.' },
]

/** Codes offered for a verdict: its own direction, plus the neutral ones. */
export function reasonCodesFor(
  decision: 'approved' | 'rejected',
  catalog: ReasonCode[] = REASON_CODES,
): ReasonCode[] {
  const wanted = decision === 'approved' ? 'approve' : 'reject'
  return catalog.filter((entry) => entry.direction === wanted || entry.direction === 'either')
}

export const CONFIDENCE_LABELS: Record<number, string> = {
  1: 'Very unsure',
  2: 'Unsure',
  3: 'Moderate',
  4: 'Confident',
  5: 'Very confident',
}
