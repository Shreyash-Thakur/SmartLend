import React, { useEffect, useState } from 'react'
import { FileClock, RefreshCw, ShieldCheck, UserCheck } from 'lucide-react'
import { Card } from '@/components/common'
import { getDecisionReport } from '@/services/applications'
import type { DecisionReport } from '@/types/api'

interface DecisionReportPanelProps {
  applicationId: string
  /** Bump to re-fetch — e.g. after a reviewer submits their decision. */
  refreshKey?: number
}

function fmt(value: number | null | undefined, digits = 4): string {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '—'
}

function fmtDuration(seconds: number | null): string {
  if (typeof seconds !== 'number' || !Number.isFinite(seconds)) return '—'
  const m = Math.floor(seconds / 60)
  const s = Math.round(seconds % 60)
  return m > 0 ? `${m}m ${s}s` : `${s}s`
}

function fmtTimestamp(value: string | null): string {
  if (!value) return '—'
  const parsed = new Date(value)
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString()
}

/**
 * The per-application audit record, rendered from
 * `GET /api/applications/{id}/report`.
 *
 * The engine half is always shown. The human half is shown only once a reviewer
 * has actually ruled; until then the panel says "awaiting human review" rather
 * than erroring, because that is the normal state of a freshly deferred case
 * and it is precisely when someone is most likely to open this report.
 */
export const DecisionReportPanel: React.FC<DecisionReportPanelProps> = ({ applicationId, refreshKey = 0 }) => {
  const [report, setReport] = useState<DecisionReport | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    setIsLoading(true)
    setError(null)
    getDecisionReport(applicationId)
      .then((result) => {
        if (cancelled) return
        setReport(result)
        if (result === null) setError('No decision report exists for this application.')
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof Error ? e.message : 'Could not load the decision report.')
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [applicationId, refreshKey])

  if (isLoading) {
    return (
      <Card title="Decision Report">
        <p className="py-8 text-center text-sm text-neutral-500 flex items-center justify-center gap-2">
          <RefreshCw className="h-4 w-4 animate-spin" /> Building the audit record…
        </p>
      </Card>
    )
  }

  if (error || !report) {
    return (
      <Card title="Decision Report">
        <p className="py-8 text-center text-sm text-neutral-500">{error ?? 'Report unavailable.'}</p>
      </Card>
    )
  }

  const { engine, humanReview } = report
  const pillars = Object.entries(engine.cbesBreakdown ?? {})
  const factors = (engine.topFactors ?? []).slice(0, 6)

  return (
    <Card
      title="Decision Report"
      description={`Audit record for ${report.applicationId} · generated ${fmtTimestamp(report.generatedAt)}`}
    >
      <div className="space-y-6">
        {/* --- engine half ------------------------------------------------ */}
        <section>
          <h4 className="flex items-center gap-2 text-sm font-semibold text-neutral-900">
            <ShieldCheck className="h-4 w-4 text-primary-500" /> Engine decision
          </h4>
          <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <Stat label="Decision" value={engine.decision} mono={false} />
            <Stat label="p(ML)" value={fmt(engine.pMl)} />
            <Stat label="p(CBES)" value={fmt(engine.pCbes)} />
            <Stat
              label="p(blend)"
              value={fmt(engine.pBlend)}
              hint={engine.pBlendSource === 'derived' ? 'recomputed — no capture row' : undefined}
            />
            <Stat label="Confidence" value={fmt(engine.confidence)} hint={engine.confidenceLabel ?? undefined} />
            <Stat label="Approve threshold" value={fmt(engine.thresholds?.approve)} />
            <Stat label="Reject threshold" value={fmt(engine.thresholds?.reject)} />
            <Stat label="Base threshold" value={fmt(engine.thresholds?.base)} />
            <Stat label="Disagreement" value={fmt(engine.disagreement)} />
            <Stat label="Risk score" value={fmt(engine.riskScore)} />
            <Stat label="Reason" value={engine.decisionReason || '—'} mono={false} />
            <Stat label="Engine version" value={engine.engineVersion ?? '—'} mono={false} />
          </div>
          {engine.explorationFlag && (
            <p className="mt-3 rounded-lg border border-blue-200 bg-blue-50 px-3 py-2 text-xs text-blue-800">
              Control-arm sample: the engine auto-decided this case but it was routed to a human anyway.
            </p>
          )}
        </section>

        {/* --- CBES pillars ----------------------------------------------- */}
        <section>
          <h4 className="text-sm font-semibold text-neutral-900">CBES pillar breakdown</h4>
          {pillars.length === 0 ? (
            <p className="mt-2 text-sm text-neutral-500">No pillar breakdown was recorded for this decision.</p>
          ) : (
            <div className="mt-3 space-y-2">
              {pillars.map(([pillar, score]) => {
                const pct = typeof score === 'number' ? Math.max(0, Math.min(1, score)) * 100 : 0
                return (
                  <div key={pillar} className="flex items-center gap-3">
                    <span className="w-32 shrink-0 text-xs capitalize text-neutral-600">{pillar}</span>
                    <div className="h-2 flex-1 overflow-hidden rounded-full bg-neutral-100">
                      <div className="h-full rounded-full bg-primary-400" style={{ width: `${pct}%` }} />
                    </div>
                    <span className="w-14 shrink-0 text-right font-mono text-xs text-neutral-700">{fmt(score, 3)}</span>
                  </div>
                )
              })}
            </div>
          )}
        </section>

        {/* --- SHAP top factors ------------------------------------------- */}
        <section>
          <h4 className="text-sm font-semibold text-neutral-900">Top contributing factors</h4>
          {factors.length === 0 ? (
            <p className="mt-2 text-sm text-neutral-500">No factor attribution is available for this decision.</p>
          ) : (
            <ul className="mt-3 space-y-1.5">
              {factors.map((factor, index) => {
                const name = String(factor.name ?? factor.feature ?? `Factor ${index + 1}`)
                const impact = typeof factor.impact === 'number' ? factor.impact : null
                return (
                  <li
                    key={`${name}-${index}`}
                    className="flex items-center justify-between gap-3 rounded-lg bg-neutral-50 px-3 py-2 text-sm"
                  >
                    <span className="min-w-0 truncate text-neutral-800">{name}</span>
                    <span
                      className={`shrink-0 font-mono text-xs ${
                        impact === null ? 'text-neutral-500' : impact >= 0 ? 'text-green-700' : 'text-red-700'
                      }`}
                    >
                      {impact === null ? '—' : `${impact >= 0 ? '+' : ''}${impact.toFixed(4)}`}
                    </span>
                  </li>
                )
              })}
            </ul>
          )}
        </section>

        {/* --- human half -------------------------------------------------- */}
        <section>
          <h4 className="flex items-center gap-2 text-sm font-semibold text-neutral-900">
            <UserCheck className="h-4 w-4 text-primary-500" /> Human review
          </h4>
          {humanReview === null ? (
            <div className="mt-3 flex items-start gap-3 rounded-xl border border-dashed border-neutral-300 bg-neutral-50 px-4 py-4 text-sm text-neutral-600">
              <FileClock className="h-5 w-5 shrink-0 text-neutral-400" />
              <span>
                {engine.routedToHumanReview
                  ? 'Awaiting human review — no reviewer has recorded a decision on this case yet.'
                  : 'This application was decided automatically and was not routed to a human reviewer.'}
              </span>
            </div>
          ) : (
            <div className="mt-3 space-y-4">
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                <Stat label="Reviewer decision" value={humanReview.decision} mono={false} />
                <Stat label="Reviewer" value={humanReview.reviewerId ?? '—'} mono={false} />
                <Stat
                  label="Reviewer confidence"
                  value={humanReview.reviewerConfidence === null ? '—' : `${humanReview.reviewerConfidence} / 5`}
                  mono={false}
                />
                <Stat label="Time on case" value={fmtDuration(humanReview.timeSpentSeconds)} mono={false} />
                <Stat label="Reviewed at" value={fmtTimestamp(humanReview.reviewedAt)} mono={false} />
                <Stat
                  label="Agreed with engine"
                  value={
                    humanReview.agreedWithEngine === null
                      ? 'No engine lean'
                      : humanReview.agreedWithEngine
                        ? 'Yes'
                        : 'No — overridden'
                  }
                  mono={false}
                />
                <Stat label="Override direction" value={humanReview.overrideDirection ?? '—'} mono={false} />
                <Stat
                  label="Outcome"
                  value={humanReview.outcomeCensored ? 'Not yet observed' : String(humanReview.realizedOutcome)}
                  mono={false}
                />
              </div>

              <div>
                <p className="text-xs font-medium uppercase tracking-wider text-neutral-500">Reasons recorded</p>
                {humanReview.reasonCodes.length === 0 ? (
                  <p className="mt-2 text-sm text-neutral-500">
                    No reason codes were recorded against this decision.
                  </p>
                ) : (
                  <div className="mt-2 flex flex-wrap gap-2">
                    {humanReview.reasonCodes.map((entry) => (
                      <span
                        key={entry.code}
                        title={`${entry.code} — ${entry.description}`}
                        className={`rounded-full border px-3 py-1 text-xs font-medium ${
                          entry.direction === 'approve'
                            ? 'border-green-200 bg-green-50 text-green-800'
                            : entry.direction === 'reject'
                              ? 'border-red-200 bg-red-50 text-red-800'
                              : 'border-neutral-200 bg-neutral-50 text-neutral-700'
                        }`}
                      >
                        {entry.label}
                      </span>
                    ))}
                  </div>
                )}
              </div>

              {humanReview.freeText && (
                <div>
                  <p className="text-xs font-medium uppercase tracking-wider text-neutral-500">Reviewer note</p>
                  <p className="mt-2 rounded-lg bg-neutral-50 px-3 py-2 text-sm text-neutral-800">
                    “{humanReview.freeText}”
                  </p>
                </div>
              )}
            </div>
          )}
        </section>
      </div>
    </Card>
  )
}

function Stat({ label, value, hint, mono = true }: { label: string; value: string; hint?: string; mono?: boolean }) {
  return (
    <div className="rounded-xl border border-neutral-200 bg-white px-3 py-2">
      <p className="text-[11px] uppercase tracking-wider text-neutral-500">{label}</p>
      <p className={`mt-1 text-sm font-semibold text-neutral-900 ${mono ? 'font-mono' : 'capitalize'}`}>{value}</p>
      {hint && <p className="mt-0.5 text-[11px] text-neutral-500">{hint}</p>}
    </div>
  )
}
