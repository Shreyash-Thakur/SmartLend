import React from 'react'
import { DashboardLayout } from '@/components/layouts/DashboardLayout'
import { Badge } from '@/components/common'
import { formatCurrency } from '@/lib/utils'
import demo from '@/data/demoTestCases.json'

type Expected = 'APPROVE' | 'DEFER' | 'REJECT'

const BADGE_STATUS: Record<Expected, 'approved' | 'deferred' | 'rejected'> = {
  APPROVE: 'approved',
  DEFER: 'deferred',
  REJECT: 'rejected',
}

const GROUPS: { expected: Expected; title: string; blurb: string }[] = [
  { expected: 'APPROVE', title: 'Confirmed approvals', blurb: 'The model is confident and the EMI is affordable.' },
  {
    expected: 'DEFER',
    title: 'Confirmed deferrals',
    blurb: 'Either the model is unsure, or it would approve a loan whose EMIs exceed income (affordability review).',
  },
  {
    expected: 'REJECT',
    title: 'Confirmed rejections',
    blurb: 'Either the model scores the customer as high risk, or the EMIs would be more than twice the income.',
  },
]

/**
 * Reference inputs for live demos. Every row is checked against the engine by
 * backend/tests/test_demo_test_cases.py, so a row here cannot silently go stale.
 */
export const DemoTestCases: React.FC = () => (
  <DashboardLayout title="Demo Test Cases">
    <section className="mb-8 rounded-2xl border border-neutral-200 bg-white p-6 shadow-[0_1px_2px_rgba(15,23,42,0.04)] sm:p-8">
      <p className="text-xs uppercase tracking-[0.28em] text-neutral-500">Demo reference</p>
      <h2 className="mt-3 text-4xl font-semibold tracking-tight text-neutral-900">Inputs with a confirmed outcome</h2>
      <p className="mt-3 max-w-3xl text-neutral-600">
        Enter these on New Application to get the stated decision. {demo.commonInputs}
      </p>
      <p className="mt-2 text-xs text-neutral-500">Verified for engine {demo.engineVersion}</p>
    </section>

    {GROUPS.map((group) => {
      const cases = demo.cases.filter((c) => c.expected === group.expected)
      return (
        <section
          key={group.expected}
          className="mb-8 rounded-2xl border border-neutral-200 bg-white p-6 shadow-[0_1px_2px_rgba(15,23,42,0.04)]"
        >
          <div className="flex flex-wrap items-center gap-3">
            <h3 className="text-xl font-semibold text-neutral-900">{group.title}</h3>
            <Badge status={BADGE_STATUS[group.expected]} />
          </div>
          <p className="mt-1 text-sm text-neutral-500">{group.blurb}</p>

          <div className="mt-4 overflow-x-auto">
            <table className="w-full min-w-[720px] text-left text-sm">
              <thead className="border-b border-neutral-200 text-neutral-500">
                <tr>
                  <th className="py-2 pr-4 font-medium">#</th>
                  <th className="py-2 pr-4 font-medium">Customer ID</th>
                  <th className="py-2 pr-4 font-medium">Loan amount</th>
                  <th className="py-2 pr-4 font-medium">Tenure</th>
                  <th className="py-2 pr-4 font-medium">Purpose</th>
                  <th className="py-2 pr-4 font-medium">Decided by</th>
                  <th className="py-2 font-medium">Why</th>
                </tr>
              </thead>
              <tbody>
                {cases.map((c) => (
                  <tr key={c.id} className="border-b border-neutral-100 align-top text-neutral-800">
                    <td className="py-3 pr-4 text-neutral-500">{c.id}</td>
                    <td className="py-3 pr-4 font-mono font-semibold">{c.customerId}</td>
                    <td className="py-3 pr-4">{formatCurrency(c.loanAmount)}</td>
                    <td className="py-3 pr-4">{c.tenureMonths} months</td>
                    <td className="py-3 pr-4 capitalize">{c.purpose.replace('_', ' ')}</td>
                    <td className="py-3 pr-4">{c.decidedBy === 'policy' ? 'Affordability policy' : 'Model'}</td>
                    <td className="py-3 text-neutral-600">{c.why}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )
    })}
  </DashboardLayout>
)
