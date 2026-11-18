import React, { useEffect } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { AlertTriangle, X } from 'lucide-react'
import { formatCurrency } from '@/lib/utils'

/**
 * The applicant's full file for the reviewer: what the bank already holds
 * (profile + bureau, merged into applicationData at submission), what the
 * applicant typed, and the affordability check the engine ran. Everything here
 * was already stored with the application; the review page simply never showed
 * it. Display only.
 */

type Tone = 'red' | 'amber' | undefined
type Row = { label: string; value: string; tone?: Tone; note?: string }

const pct = (value: number) => `${(value * 100).toFixed(1)}%`

function pick(data: Record<string, unknown>, ...keys: string[]): unknown {
  for (const key of keys) {
    const value = data[key]
    if (value !== undefined && value !== null && value !== '') return value
  }
  return undefined
}

function num(data: Record<string, unknown>, ...keys: string[]): number | undefined {
  const value = Number(pick(data, ...keys))
  return Number.isFinite(value) ? value : undefined
}

function text(data: Record<string, unknown>, ...keys: string[]): string {
  const value = pick(data, ...keys)
  if (value === undefined) return '—'
  if (typeof value === 'boolean') return value ? 'Yes' : 'No'
  if (Array.isArray(value)) return value.length ? value.join(', ') : 'None'
  return String(value).replace(/_/g, ' ')
}

function buildFile(data: Record<string, unknown>) {
  const meta = (data._decision_meta ?? {}) as Record<string, unknown>
  // The bank's own record wins over form fields with similar names (the form's
  // default marital status once masked the profile's).
  const bankRecord = { ...data, ...((data.customer_profile ?? {}) as Record<string, unknown>) }
  const affordability = (meta.affordability ?? null) as { foir?: number; newEmi?: number; action?: string | null } | null

  const cibil = num(data, 'cibil_score', 'cibilScore')
  const missed = num(data, 'missed_payments', 'missedPayments', 'delinquencies')
  const active = num(data, 'active_loans', 'activeLoans')
  const closed = num(data, 'closed_loans', 'closedLoans')
  const total = num(data, 'total_loans', 'totalLoans')
  const utilisation = num(data, 'credit_utilization_ratio', 'creditUtilizationRatio')
  const annualIncome = num(data, 'annual_income', 'annualIncome') ?? 0
  const monthlyIncome = num(data, 'monthly_income', 'monthlyIncome') ?? annualIncome / 12
  // existing_emis is annual in the profile (DTI = existing_emis / annual_income).
  const existingMonthly = (num(data, 'existing_emis', 'existingEmis') ?? 0) / 12
  const otherBank = num(data, 'obligations_other_banks', 'emi') ?? 0
  const additionalIncome = num(data, 'additional_income') ?? 0
  const years = num(data, 'employment_tenure_years', 'years_employed', 'yearsOfEmployment')
  const loanAmount = num(data, 'loanAmount', 'loan_amount') ?? 0
  const loanToIncome = annualIncome > 0 ? loanAmount / annualIncome : undefined
  const newEmi = affordability?.newEmi
  const foir = affordability?.foir

  const bank: Row[] = [
    {
      label: 'CIBIL score',
      value: cibil !== undefined ? String(Math.round(cibil)) : '—',
      tone: cibil === undefined ? undefined : cibil < 650 ? 'red' : cibil < 750 ? 'amber' : undefined,
      note: cibil !== undefined && cibil < 650 ? 'Below the 650 most lenders want for unsecured loans' : undefined,
    },
    {
      label: 'Missed payments',
      value: missed !== undefined ? String(missed) : '—',
      tone: missed && missed > 0 ? 'red' : undefined,
    },
    {
      label: 'Loans: total / active / closed',
      value: `${total ?? '—'} / ${active ?? '—'} / ${closed ?? '—'}`,
      tone: (active ?? 0) >= 3 || ((active ?? 0) > 0 && closed === 0) ? 'amber' : undefined,
      note: (active ?? 0) > 0 && closed === 0 ? 'No loan repaid in full yet' : undefined,
    },
    {
      label: 'Credit utilisation',
      value: utilisation !== undefined ? pct(utilisation) : '—',
      tone: utilisation === undefined ? undefined : utilisation > 0.75 ? 'red' : utilisation > 0.5 ? 'amber' : undefined,
    },
    { label: 'Existing EMIs with us', value: `${formatCurrency(existingMonthly)} / month` },
    { label: 'Income', value: `${formatCurrency(monthlyIncome)} / month (${formatCurrency(annualIncome)} / year)` },
    {
      label: 'Employment',
      value: `${text(bankRecord, 'employment_type', 'employmentType')}${years !== undefined ? ` · ${years.toFixed(1)} years` : ''}`,
      tone: years !== undefined && years < 2 ? 'amber' : undefined,
    },
    { label: 'Age · dependents · marital status', value: `${text(bankRecord, 'age')} · ${text(bankRecord, 'dependents')} · ${text(bankRecord, 'marital_status', 'maritalStatus')}` },
    { label: 'Region', value: `${text(bankRecord, 'regionLabel', 'region')} (rating ${text(bankRecord, 'region_rating')})` },
    { label: 'Products held with us', value: text(data, 'relationship_products') },
  ]

  const applied: Row[] = [
    { label: 'Amount · tenure · rate', value: `${formatCurrency(loanAmount)} · ${text(data, 'loanTenure', 'loan_tenure_months')} months · ${text(data, 'interestRate')}%` },
    {
      label: 'Loan vs yearly income',
      value: loanToIncome !== undefined ? `${loanToIncome.toFixed(2)}×` : '—',
      tone: loanToIncome !== undefined && loanToIncome > 1 ? 'amber' : undefined,
    },
    { label: 'Purpose (category)', value: text(data, 'loanPurpose', 'loan_purpose') },
    { label: 'Purpose (in their words)', value: text(data, 'loan_purpose_details') },
    { label: 'Repayment source', value: text(data, 'repayment_source') },
    { label: 'EMIs at other lenders', value: `${formatCurrency(otherBank)} / month` },
    { label: 'Additional income', value: `${formatCurrency(additionalIncome)} / month` },
    { label: 'Collateral', value: `${text(data, 'collateral_type')}${num(data, 'collateral_value') ? ` · ${formatCurrency(num(data, 'collateral_value') ?? 0)}` : ''}` },
    { label: 'Guarantor', value: pick(data, 'guarantor_customer_id') ? `${text(data, 'guarantor_customer_id')} (${text(data, 'guarantor_relationship')})` : 'None' },
    {
      label: 'Job change in last 6 months',
      value: text(data, 'employment_changed_recently'),
      tone: pick(data, 'employment_changed_recently') === true ? 'amber' : undefined,
    },
  ]

  const affordabilityRows: Row[] =
    foir === undefined
      ? [{ label: 'Affordability check', value: 'Not run — this application predates the FOIR policy' }]
      : [
          { label: 'New EMI for this loan', value: newEmi !== undefined ? `${formatCurrency(newEmi)} / month` : '—' },
          {
            label: 'All EMIs after this loan',
            value: `${formatCurrency(existingMonthly + otherBank + (newEmi ?? 0))} / month`,
          },
          { label: 'Income available', value: `${formatCurrency(monthlyIncome + additionalIncome)} / month` },
          {
            label: 'FOIR (EMIs ÷ income)',
            value: pct(foir),
            tone: foir > 1 ? 'red' : foir > 0.5 ? 'amber' : undefined,
            note: 'Policy: above 100% goes to review, above 200% is rejected. Many lenders cap at 50–60%.',
          },
          {
            label: 'Policy result',
            value: affordability?.action ? `${affordability.action} (affordability policy)` : 'Passed — no policy action',
            tone: affordability?.action === 'REJECT' ? 'red' : affordability?.action ? 'amber' : undefined,
          },
        ]

  const flags = [...bank, ...applied, ...affordabilityRows].filter((row) => row.tone)
  return { bank, applied, affordability: affordabilityRows, flags }
}

const TONE_CLASS: Record<string, string> = {
  red: 'text-red-700',
  amber: 'text-amber-700',
}

const Section: React.FC<{ title: string; subtitle: string; rows: Row[] }> = ({ title, subtitle, rows }) => (
  <section>
    <h3 className="text-sm font-semibold text-neutral-900">{title}</h3>
    <p className="mb-2 text-xs text-neutral-500">{subtitle}</p>
    <dl className="divide-y divide-neutral-100 rounded-xl border border-neutral-200">
      {rows.map((row) => (
        <div key={row.label} className="grid grid-cols-5 gap-3 px-3 py-2 text-sm">
          <dt className="col-span-2 text-neutral-500">{row.label}</dt>
          <dd className={`col-span-3 font-medium ${row.tone ? TONE_CLASS[row.tone] : 'text-neutral-900'}`}>
            {row.value}
            {row.note && <span className="block text-xs font-normal text-neutral-500">{row.note}</span>}
          </dd>
        </div>
      ))}
    </dl>
  </section>
)

export const ApplicantFileDrawer: React.FC<{
  open: boolean
  onClose: () => void
  applicantName: string
  data: Record<string, unknown>
}> = ({ open, onClose, applicantName, data }) => {
  useEffect(() => {
    if (!open) return
    const onKey = (event: KeyboardEvent) => event.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, onClose])

  const file = buildFile(data)

  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            className="fixed inset-0 z-40 bg-black/40"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
          />
          <motion.aside
            role="dialog"
            aria-label="Applicant file"
            className="fixed inset-y-0 right-0 z-50 flex w-full max-w-xl flex-col bg-white shadow-2xl"
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ type: 'tween', duration: 0.22 }}
          >
            <header className="flex items-center justify-between border-b border-neutral-200 px-5 py-4">
              <div>
                <p className="text-xs uppercase tracking-wide text-neutral-500">Applicant file</p>
                <h2 className="text-lg font-semibold text-neutral-900">{applicantName}</h2>
              </div>
              <button type="button" onClick={onClose} aria-label="Close" className="rounded-lg p-2 text-neutral-500 hover:bg-neutral-100">
                <X className="h-5 w-5" />
              </button>
            </header>

            <div className="flex-1 space-y-6 overflow-y-auto px-5 py-5">
              {file.flags.length > 0 && (
                <div className="rounded-xl border border-amber-200 bg-amber-50 p-3">
                  <p className="mb-1 flex items-center gap-2 text-sm font-semibold text-amber-900">
                    <AlertTriangle className="h-4 w-4" /> Worth checking ({file.flags.length})
                  </p>
                  <ul className="list-disc space-y-0.5 pl-6 text-sm text-amber-900">
                    {file.flags.map((row) => (
                      <li key={row.label}>
                        {row.label}: {row.value}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              <Section title="Bank record" subtitle="Already held by the bank: account history and credit bureau." rows={file.bank} />
              <Section title="What they applied for" subtitle="Typed by the applicant on the form." rows={file.applied} />
              <Section title="Affordability" subtitle="The FOIR check the engine ran on this application." rows={file.affordability} />
            </div>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  )
}
