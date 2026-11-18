import React from 'react'
import { CheckCircle2, Circle, Clock, XCircle } from 'lucide-react'
import { Card } from '@/components/common'
import type { LoanApplication } from '@/types/application'

type StepState = 'done' | 'current' | 'pending' | 'rejected'

interface Step {
  title: string
  detail?: string
  at?: string | null
  state: StepState
}

const RECOMMENDATION_LABEL: Record<string, string> = {
  approved: 'approve',
  rejected: 'reject',
  deferred: 'send to a human reviewer',
}

/** Where the application stands, built only from fields the API already
 *  returns: createdAt, modelRecommendation, routedToHumanReview,
 *  manualDecisionApplied/manualDecidedAt and the final status. */
function buildSteps(application: LoanApplication, audience: 'customer' | 'org'): Step[] {
  const recommendation = application.modelRecommendation ?? application.status
  const humanReview = Boolean(
    application.manualDecisionApplied || application.routedToHumanReview || recommendation === 'deferred',
  )
  const final = application.status === 'approved' || application.status === 'rejected'
  const steps: Step[] = [
    { title: 'Application submitted', at: application.createdAt, state: 'done' },
    {
      title: 'Automated check complete',
      detail:
        audience === 'org'
          ? `Model recommended: ${RECOMMENDATION_LABEL[recommendation] ?? recommendation}`
          : humanReview
            ? 'Your application needs a closer look by our team'
            : 'Checked against your profile and our lending policy',
      at: application.createdAt,
      state: 'done',
    },
  ]

  if (humanReview) {
    steps.push({
      title: application.manualDecisionApplied ? 'Reviewed by a loan officer' : 'With a loan officer for review',
      detail: application.manualDecisionApplied ? undefined : 'Usually within 1–2 working days',
      at: application.manualDecidedAt,
      state: application.manualDecisionApplied ? 'done' : 'current',
    })
  }

  steps.push({
    title: final ? `Decision: ${application.status === 'approved' ? 'Approved' : 'Rejected'}` : 'Final decision',
    detail: final ? (application.manualDecisionApplied ? 'Decided by a loan officer' : 'Decided automatically') : undefined,
    at: final ? application.manualDecidedAt ?? application.createdAt : null,
    state: final ? (application.status === 'approved' ? 'done' : 'rejected') : 'pending',
  })
  return steps
}

/** createdAt arrives as naive UTC ("2026-11-18T04:38:53"); read it as UTC, not local. */
function formatTime(iso: string): string {
  const hasZone = /(?:[zZ]|[+-]\d{2}:?\d{2})$/.test(iso)
  return new Date(hasZone ? iso : `${iso}Z`).toLocaleString()
}

const ICONS: Record<StepState, React.ReactNode> = {
  done: <CheckCircle2 className="h-5 w-5 text-green-600" />,
  rejected: <XCircle className="h-5 w-5 text-red-600" />,
  current: <Clock className="h-5 w-5 text-amber-500" />,
  pending: <Circle className="h-5 w-5 text-neutral-300" />,
}

export const StatusTimeline: React.FC<{ application: LoanApplication; audience: 'customer' | 'org' }> = ({
  application,
  audience,
}) => {
  const steps = buildSteps(application, audience)
  return (
    <Card title="Application Status">
      <ol data-i18n-live className="space-y-0">
        {steps.map((step, index) => (
          <li key={step.title} className="relative flex gap-3 pb-5 last:pb-0">
            {index < steps.length - 1 && (
              <span className="absolute left-[9px] top-6 h-[calc(100%-1rem)] w-px bg-neutral-200" aria-hidden />
            )}
            <span className="relative z-10 mt-0.5 shrink-0">{ICONS[step.state]}</span>
            <div className="min-w-0">
              <p className={`text-sm font-medium ${step.state === 'pending' ? 'text-neutral-400' : 'text-neutral-900'}`}>
                {step.title}
              </p>
              {step.detail && <p className="text-xs text-neutral-500">{step.detail}</p>}
              {step.at && <p className="text-xs text-neutral-400">{formatTime(step.at)}</p>}
            </div>
          </li>
        ))}
      </ol>
    </Card>
  )
}
