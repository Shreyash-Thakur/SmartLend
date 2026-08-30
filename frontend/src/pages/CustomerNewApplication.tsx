import React, { useCallback, useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { AlertCircle, ChevronLeft, Landmark, ShieldCheck, UserCheck, Wallet } from 'lucide-react'
import { DashboardLayout } from '@/components/layouts/DashboardLayout'
import { Button, Card, Input, Select } from '@/components/common'
import { useApplicationData } from '@/hooks/useApplicationData'
import { useAuth } from '@/hooks/useAuth'
import { getCustomerProfile } from '@/services/applications'
import { trackEvent } from '@/services/analytics'
import type {
  CollateralType,
  CustomerProfile,
  GuarantorRelationship,
  LoanPurposeShort,
  RepaymentSource,
  ShortLoanApplicationFormData,
} from '@/types/application'

// Mirrors backend/app/schemas.py GUARANTOR_THRESHOLD. The backend recomputes
// `guarantor_required` itself and never trusts the client; this copy only
// decides whether to *show* the annexure.
const GUARANTOR_THRESHOLD = 500_000

const LOAN_PURPOSE_OPTIONS = [
  { value: '', label: 'Select...' },
  { value: 'home_improvement', label: 'Home improvement' },
  { value: 'education', label: 'Education' },
  { value: 'medical', label: 'Medical' },
  { value: 'debt_consolidation', label: 'Debt consolidation' },
  { value: 'business', label: 'Business' },
  { value: 'vehicle', label: 'Vehicle' },
  { value: 'wedding', label: 'Wedding' },
  { value: 'other', label: 'Other' },
]

const TENURE_OPTIONS = [12, 24, 36, 48, 60].map((months) => ({
  value: String(months),
  label: `${months} months`,
}))

const REPAYMENT_SOURCE_OPTIONS = [
  { value: 'salary', label: 'Salary' },
  { value: 'business_income', label: 'Business income' },
  { value: 'rental', label: 'Rental income' },
  { value: 'other', label: 'Other' },
]

const EMI_DATE_OPTIONS = [1, 5, 10].map((day) => ({ value: String(day), label: `${day}th of the month` }))

const COLLATERAL_OPTIONS = [
  { value: 'none', label: 'None (unsecured)' },
  { value: 'property', label: 'Property' },
  { value: 'gold', label: 'Gold' },
  { value: 'fixed_deposit', label: 'Fixed deposit' },
  { value: 'vehicle', label: 'Vehicle' },
]

const GUARANTOR_RELATIONSHIP_OPTIONS = [
  { value: '', label: 'Select...' },
  { value: 'spouse', label: 'Spouse' },
  { value: 'parent', label: 'Parent' },
  { value: 'sibling', label: 'Sibling' },
  { value: 'employer', label: 'Employer' },
  { value: 'other', label: 'Other' },
]

interface FormState {
  customerId: string
  loanAmount: string
  loanPurpose: LoanPurposeShort | ''
  loanTenureMonths: string
  repaymentSource: RepaymentSource
  preferredEmiDate: string
  endUseDeclaration: boolean
  obligationsOtherBanks: string
  employmentChangedRecently: boolean
  additionalIncome: string
  collateralType: CollateralType
  collateralValue: string
  isRelatedParty: boolean
  guarantorRelationship: GuarantorRelationship | ''
  guarantorCustomerId: string
}

const initialState: FormState = {
  customerId: '',
  loanAmount: '',
  loanPurpose: '',
  loanTenureMonths: '36',
  repaymentSource: 'salary',
  preferredEmiDate: '5',
  endUseDeclaration: false,
  obligationsOtherBanks: '',
  employmentChangedRecently: false,
  additionalIncome: '',
  collateralType: 'none',
  collateralValue: '',
  isRelatedParty: false,
  guarantorRelationship: '',
  guarantorCustomerId: '',
}

function toNumber(value: string): number {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : 0
}

function formatCurrency(value: number | null | undefined) {
  if (value === null || value === undefined) return 'Not on file'
  return `₹${Math.round(value).toLocaleString()}`
}

function formatText(value: string | number | null | undefined) {
  if (value === null || value === undefined || value === '') return 'Not on file'
  return String(value)
}

function formatPercent(value: number | null | undefined) {
  if (value === null || value === undefined) return 'Not on file'
  return `${(value * 100).toFixed(1)}%`
}

export const CustomerNewApplication: React.FC = () => {
  const navigate = useNavigate()
  const { user } = useAuth()
  const { addApplication, isLoading, error } = useApplicationData({ scope: 'customer', applicantId: user?.uid })

  const [form, setForm] = useState<FormState>(initialState)
  const [profile, setProfile] = useState<CustomerProfile | null>(null)
  const [profileState, setProfileState] = useState<'idle' | 'loading' | 'found' | 'missing' | 'error'>('idle')
  const [submitError, setSubmitError] = useState<string | null>(null)
  const [isSubmitting, setIsSubmitting] = useState(false)

  const setField = useCallback(<K extends keyof FormState>(key: K, value: FormState[K]) => {
    setForm((current) => ({ ...current, [key]: value }))
  }, [])

  // Pull the profile as soon as a plausible customer id has been typed, so the
  // "we already have this" panel fills in before the applicant answers
  // anything. Debounced to avoid a request per keystroke.
  useEffect(() => {
    const customerId = form.customerId.trim()
    if (!customerId) {
      setProfile(null)
      setProfileState('idle')
      return
    }

    let cancelled = false
    setProfileState('loading')
    const timer = window.setTimeout(() => {
      getCustomerProfile(customerId)
        .then((result) => {
          if (cancelled) return
          setProfile(result)
          setProfileState(result ? 'found' : 'missing')
        })
        .catch(() => {
          if (cancelled) return
          setProfile(null)
          setProfileState('error')
        })
    }, 400)

    return () => {
      cancelled = true
      window.clearTimeout(timer)
    }
  }, [form.customerId])

  const loanAmount = toNumber(form.loanAmount)
  const guarantorRequired = form.collateralType === 'none' && loanAmount > GUARANTOR_THRESHOLD

  const canSubmit =
    profileState === 'found'
    && loanAmount > 0
    && form.loanPurpose !== ''
    && form.endUseDeclaration
    && (!guarantorRequired || form.guarantorRelationship !== '')

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault()
    if (!canSubmit || isSubmitting) return
    setSubmitError(null)
    setIsSubmitting(true)

    const payload: ShortLoanApplicationFormData = {
      applicantId: user?.uid,
      customer_id: form.customerId.trim(),
      loan_amount: loanAmount,
      loan_purpose: form.loanPurpose as LoanPurposeShort,
      loan_tenure_months: toNumber(form.loanTenureMonths),
      repayment_source: form.repaymentSource,
      preferred_emi_date: toNumber(form.preferredEmiDate),
      end_use_declaration: form.endUseDeclaration,
      obligations_other_banks: toNumber(form.obligationsOtherBanks),
      employment_changed_recently: form.employmentChangedRecently,
      additional_income: toNumber(form.additionalIncome),
      collateral_type: form.collateralType,
      collateral_value: form.collateralType === 'none' ? 0 : toNumber(form.collateralValue),
      is_related_party: form.isRelatedParty,
      ...(guarantorRequired
        ? {
            guarantor_relationship: form.guarantorRelationship as GuarantorRelationship,
            guarantor_customer_id: form.guarantorCustomerId.trim() || undefined,
          }
        : {}),
    }

    try {
      const application = await addApplication(payload)
      trackEvent('application_submitted', { applicationId: application.id })
      navigate('/dashboard/customer?view=history')
    } catch (submitFailure) {
      setSubmitError(submitFailure instanceof Error ? submitFailure.message : 'Failed to submit application')
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <DashboardLayout title="Customer Dashboard" role="customer">
      <section className="mb-6">
        <Button
          variant="ghost"
          leftIcon={<ChevronLeft className="h-4 w-4" />}
          onClick={() => navigate('/dashboard/customer')}
        >
          Back to dashboard
        </Button>
      </section>

      {(error || submitError) && (
        <section className="mb-6">
          <Card className="border-red-200 bg-red-50">
            <p className="text-red-700">{submitError ?? error}</p>
          </Card>
        </section>
      )}

      <section>
        <Card
          title="New Loan Application"
          description="You are an existing customer, so we only ask for what we cannot already see. Everything else is pulled from your account and your bureau record."
          className="rounded-[36px] border-white/80"
        >
          <form onSubmit={handleSubmit} className="space-y-8">
            {/* ---- Customer identification + read-only pulled profile ---- */}
            <section className="rounded-[32px] border border-white/70 bg-white/85 p-6 shadow-xl backdrop-blur-xl">
              <div className="mb-5 flex items-center justify-between">
                <div>
                  <p className="text-xs uppercase tracking-[0.24em] text-neutral-500">Your Account</p>
                  <h3 className="mt-2 text-2xl font-semibold text-neutral-900">Customer Identification</h3>
                </div>
                <div className="rounded-2xl bg-primary-50 p-3 text-primary-600">
                  <UserCheck className="h-6 w-6" />
                </div>
              </div>

              <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
                <Input
                  label="Customer ID *"
                  required
                  value={form.customerId}
                  onChange={(event) => setField('customerId', event.target.value)}
                  hint="Your existing bank customer number"
                  error={
                    profileState === 'missing'
                      ? 'No customer record found for this ID'
                      : profileState === 'error'
                        ? 'Could not reach the customer record service'
                        : undefined
                  }
                />
              </div>

              {profileState === 'loading' && (
                <p className="mt-4 text-sm text-neutral-500">Looking up your record…</p>
              )}

              {profileState === 'found' && profile && (
                <div className="mt-6 rounded-[28px] border border-primary-100 bg-primary-50/60 p-5">
                  <div className="flex items-center gap-2">
                    <ShieldCheck className="h-5 w-5 text-primary-600" />
                    <p className="text-sm font-semibold text-primary-900">
                      We already have this — no need to re-enter it
                    </p>
                  </div>
                  <p className="mt-1 text-xs text-primary-800/80">
                    Read-only. Pulled from your KYC record, your account history and your bureau report.
                  </p>

                  <div className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-3">
                    <ProfileBlock
                      title="KYC & demographics"
                      rows={[
                        ['Age', formatText(profile.age)],
                        ['Gender', formatText(profile.gender)],
                        ['Dependents', formatText(profile.dependents)],
                        ['Region', formatText(profile.region)],
                        ['Employment', formatText(profile.employment_type)],
                        ['Employment tenure', profile.employment_tenure_years === null
                          ? 'Not on file'
                          : `${profile.employment_tenure_years} years`],
                      ]}
                    />
                    <ProfileBlock
                      title="Account history"
                      rows={[
                        ['Annual income', formatCurrency(profile.annual_income)],
                        ['Monthly income', formatCurrency(profile.monthly_income)],
                        ['Existing EMIs (with us)', formatCurrency(profile.existing_emis)],
                        ['Debt-to-income', formatPercent(profile.dti)],
                      ]}
                    />
                    <ProfileBlock
                      title="Bureau pull"
                      rows={[
                        ['Credit score', formatText(profile.credit_score_display)],
                        ['Delinquencies', formatText(profile.delinquencies)],
                        ['Active loans', formatText(profile.active_loans)],
                        ['Closed loans', formatText(profile.closed_loans)],
                        ['Credit utilisation', formatPercent(profile.credit_utilisation)],
                      ]}
                      footnote={profile.credit_score_basis ?? undefined}
                    />
                  </div>
                </div>
              )}
            </section>

            {/* ---- Section A · Loan request ---- */}
            <section className="rounded-[32px] border border-white/70 bg-white/85 p-6 shadow-xl backdrop-blur-xl">
              <div className="mb-5 flex items-center justify-between">
                <div>
                  <p className="text-xs uppercase tracking-[0.24em] text-neutral-500">Section A</p>
                  <h3 className="mt-2 text-2xl font-semibold text-neutral-900">Loan Request</h3>
                </div>
                <div className="rounded-2xl bg-accent-50 p-3 text-accent-500">
                  <Wallet className="h-6 w-6" />
                </div>
              </div>

              <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
                <Input
                  label="Loan Amount *"
                  type="number"
                  required
                  value={form.loanAmount}
                  onChange={(event) => setField('loanAmount', event.target.value)}
                />

                <Select
                  label="Loan Purpose *"
                  options={LOAN_PURPOSE_OPTIONS}
                  value={form.loanPurpose}
                  onChange={(value) => setField('loanPurpose', String(value) as LoanPurposeShort)}
                />

                <Select
                  label="Tenure *"
                  options={TENURE_OPTIONS}
                  value={form.loanTenureMonths}
                  onChange={(value) => setField('loanTenureMonths', String(value))}
                />

                <Select
                  label="Repayment Source *"
                  options={REPAYMENT_SOURCE_OPTIONS}
                  value={form.repaymentSource}
                  onChange={(value) => setField('repaymentSource', String(value) as RepaymentSource)}
                />

                <Select
                  label="Preferred EMI Date"
                  options={EMI_DATE_OPTIONS}
                  value={form.preferredEmiDate}
                  onChange={(value) => setField('preferredEmiDate', String(value))}
                />
              </div>

              <label className="mt-5 flex cursor-pointer items-start gap-3 rounded-2xl border border-neutral-200 bg-neutral-50 p-4">
                <input
                  type="checkbox"
                  className="mt-1 h-4 w-4 rounded border-neutral-300"
                  checked={form.endUseDeclaration}
                  onChange={(event) => setField('endUseDeclaration', event.target.checked)}
                />
                <span className="text-sm text-neutral-700">
                  <span className="font-medium text-neutral-900">End-use declaration (required).</span>{' '}
                  I confirm the funds will not be used for speculative or anti-social purposes.
                </span>
              </label>
            </section>

            {/* ---- Section B · What the bank cannot see ---- */}
            <section className="rounded-[32px] border border-white/70 bg-white/85 p-6 shadow-xl backdrop-blur-xl">
              <div className="mb-5 flex items-center justify-between">
                <div>
                  <p className="text-xs uppercase tracking-[0.24em] text-neutral-500">Section B</p>
                  <h3 className="mt-2 text-2xl font-semibold text-neutral-900">What We Cannot See</h3>
                  <p className="mt-1 text-sm text-neutral-500">
                    We can see our own accounts and your bureau report, but bureau data lags 30–45 days.
                  </p>
                </div>
                <div className="rounded-2xl bg-secondary-50 p-3 text-secondary-600">
                  <Landmark className="h-6 w-6" />
                </div>
              </div>

              <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
                <Input
                  label="Monthly EMIs at other lenders"
                  type="number"
                  value={form.obligationsOtherBanks}
                  onChange={(event) => setField('obligationsOtherBanks', event.target.value)}
                  hint="Obligations we cannot observe"
                />

                <Input
                  label="Additional monthly income"
                  type="number"
                  value={form.additionalIncome}
                  onChange={(event) => setField('additionalIncome', event.target.value)}
                  hint="Rental, freelance, spouse income"
                />

                <Select
                  label="Collateral offered"
                  options={COLLATERAL_OPTIONS}
                  value={form.collateralType}
                  onChange={(value) => setField('collateralType', String(value) as CollateralType)}
                />

                {form.collateralType !== 'none' && (
                  <Input
                    label="Collateral value"
                    type="number"
                    value={form.collateralValue}
                    onChange={(event) => setField('collateralValue', event.target.value)}
                  />
                )}
              </div>

              <div className="mt-5 space-y-3">
                <label className="flex cursor-pointer items-start gap-3 rounded-2xl border border-neutral-200 bg-neutral-50 p-4">
                  <input
                    type="checkbox"
                    className="mt-1 h-4 w-4 rounded border-neutral-300"
                    checked={form.employmentChangedRecently}
                    onChange={(event) => setField('employmentChangedRecently', event.target.checked)}
                  />
                  <span className="text-sm text-neutral-700">
                    I have changed jobs in the last 6 months.
                  </span>
                </label>

                <label className="flex cursor-pointer items-start gap-3 rounded-2xl border border-neutral-200 bg-neutral-50 p-4">
                  <input
                    type="checkbox"
                    className="mt-1 h-4 w-4 rounded border-neutral-300"
                    checked={form.isRelatedParty}
                    onChange={(event) => setField('isRelatedParty', event.target.checked)}
                  />
                  <span className="text-sm text-neutral-700">
                    <span className="font-medium text-neutral-900">Related-party declaration (RBI).</span>{' '}
                    I am a director of a bank, or a relative of one.
                  </span>
                </label>
              </div>
            </section>

            {/* ---- Section C · Guarantor (conditional annexure) ---- */}
            {guarantorRequired && (
              <section className="rounded-[32px] border border-white/70 bg-white/85 p-6 shadow-xl backdrop-blur-xl">
                <div className="mb-5">
                  <p className="text-xs uppercase tracking-[0.24em] text-neutral-500">Section C</p>
                  <h3 className="mt-2 text-2xl font-semibold text-neutral-900">Guarantor</h3>
                  <p className="mt-1 text-sm text-neutral-500">
                    Required because this request is unsecured and above ₹{GUARANTOR_THRESHOLD.toLocaleString()}.
                  </p>
                </div>

                <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
                  <Select
                    label="Relationship to you *"
                    options={GUARANTOR_RELATIONSHIP_OPTIONS}
                    value={form.guarantorRelationship}
                    onChange={(value) => setField('guarantorRelationship', String(value) as GuarantorRelationship)}
                  />

                  <Input
                    label="Guarantor customer ID"
                    value={form.guarantorCustomerId}
                    onChange={(event) => setField('guarantorCustomerId', event.target.value)}
                    hint="If they bank with us, we pull their profile instead of re-keying it"
                  />
                </div>
              </section>
            )}

            {profileState !== 'found' && (
              <div className="rounded-lg border-l-4 border-blue-500 bg-blue-50 p-4">
                <div className="flex gap-3">
                  <AlertCircle className="mt-0.5 h-5 w-5 flex-shrink-0 text-blue-600" />
                  <div>
                    <p className="font-medium text-blue-900">Enter your customer ID to continue</p>
                    <p className="mt-1 text-sm text-blue-800">
                      We score the application against the record we already hold on you, so the ID must resolve
                      before it can be submitted.
                    </p>
                  </div>
                </div>
              </div>
            )}

            <Button
              variant="primary"
              size="lg"
              type="submit"
              isLoading={isSubmitting || isLoading}
              fullWidth
              disabled={!canSubmit}
              className="rounded-2xl py-4 shadow-lg"
            >
              {isSubmitting || isLoading ? 'Submitting...' : 'Submit Application'}
            </Button>
          </form>
        </Card>
      </section>
    </DashboardLayout>
  )
}

function ProfileBlock({
  title,
  rows,
  footnote,
}: {
  title: string
  rows: Array<[string, string]>
  footnote?: string
}) {
  return (
    <div className="rounded-2xl border border-neutral-200 bg-white p-4">
      <p className="text-xs uppercase tracking-[0.18em] text-neutral-500">{title}</p>
      <div className="mt-3 space-y-2 text-sm text-neutral-700">
        {rows.map(([label, value]) => (
          <p key={label}>
            <span className="font-medium text-neutral-900">{label}:</span> {value}
          </p>
        ))}
      </div>
      {footnote && <p className="mt-3 text-xs text-neutral-400">{footnote}</p>}
    </div>
  )
}
