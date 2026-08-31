import React, { useCallback, useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  AlertCircle,
  CheckCircle2,
  ChevronLeft,
  FileText,
  Gem,
  Landmark,
  Mic,
  ShieldCheck,
  Square,
  UserCheck,
  Users,
  Volume2,
  Wallet,
} from 'lucide-react'
import { DashboardLayout } from '@/components/layouts/DashboardLayout'
import { Button, Card, Input, Select, Textarea } from '@/components/common'
import { useApplicationData } from '@/hooks/useApplicationData'
import { useAuth } from '@/hooks/useAuth'
import { getCustomerProfile, getCustomerSamples } from '@/services/applications'
import { trackEvent } from '@/services/analytics'
import {
  getVoiceStatus,
  isRecordingSupported,
  synthesizeDecision,
  transcribeAudio,
  type VoiceStatus,
} from '@/services/voice'
import type {
  AdditionalIncomeType,
  CollateralType,
  CustomerProfile,
  CustomerSample,
  DisbursementAccount,
  EmiStartPreference,
  GuarantorRelationship,
  LoanApplication,
  LoanPurposeShort,
  OwnershipProofType,
  PrepaymentIntent,
  RelationshipProduct,
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

const TENURE_OPTIONS = [12, 24, 36, 48, 60, 84].map((months) => ({
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

// Disbursement account, moratorium and prepayment intent all appear on the SBI
// and HDFC forms cited in docs/FORM-REDESIGN.md. None of them are inferable:
// they are the applicant's *preference*, not a fact on file.
const DISBURSEMENT_ACCOUNT_OPTIONS = [
  { value: 'primary_savings', label: 'Primary savings account' },
  { value: 'salary_account', label: 'Salary account' },
  { value: 'current_account', label: 'Current account' },
  { value: 'new_account', label: 'Open a new account' },
]

const EMI_START_OPTIONS = [
  { value: 'next_cycle', label: 'Next billing cycle' },
  { value: 'after_30_days', label: 'After 30 days' },
  { value: 'after_60_days', label: 'After 60 days (moratorium)' },
  { value: 'after_90_days', label: 'After 90 days (moratorium)' },
]

const PREPAYMENT_OPTIONS = [
  { value: 'none', label: 'No prepayment planned' },
  { value: 'partial_within_12m', label: 'Partial, within 12 months' },
  { value: 'full_within_24m', label: 'Full closure, within 24 months' },
  { value: 'undecided', label: 'Undecided' },
]

const ADDITIONAL_INCOME_TYPE_OPTIONS = [
  { value: '', label: 'Select...' },
  { value: 'rental', label: 'Rental' },
  { value: 'freelance', label: 'Freelance / consulting' },
  { value: 'spouse', label: 'Spouse income' },
  { value: 'investments', label: 'Investments / dividends' },
  { value: 'pension', label: 'Pension' },
  { value: 'agriculture', label: 'Agriculture' },
  { value: 'other', label: 'Other' },
]

const RELATIONSHIP_PRODUCT_OPTIONS: Array<{ value: RelationshipProduct; label: string }> = [
  { value: 'savings', label: 'Savings account' },
  { value: 'current', label: 'Current account' },
  { value: 'fixed_deposit', label: 'Fixed deposit' },
  { value: 'credit_card', label: 'Credit card' },
  { value: 'demat', label: 'Demat account' },
  { value: 'insurance', label: 'Insurance policy' },
  { value: 'locker', label: 'Safe deposit locker' },
]

const COLLATERAL_OPTIONS = [
  { value: 'none', label: 'None (unsecured)' },
  { value: 'property', label: 'Property' },
  { value: 'gold', label: 'Gold' },
  { value: 'fixed_deposit', label: 'Fixed deposit' },
  { value: 'vehicle', label: 'Vehicle' },
]

const OWNERSHIP_PROOF_OPTIONS = [
  { value: '', label: 'Select...' },
  { value: 'sale_deed', label: 'Sale deed' },
  { value: 'registry_extract', label: 'Registry extract / 7-12' },
  { value: 'rc_book', label: 'RC book' },
  { value: 'fd_receipt', label: 'FD receipt' },
  { value: 'gold_appraisal', label: 'Gold appraisal certificate' },
  { value: 'other', label: 'Other' },
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

  // Section A · Loan request
  loanAmount: string
  loanPurpose: LoanPurposeShort | ''
  loanPurposeDetails: string
  loanTenureMonths: string
  repaymentSource: RepaymentSource
  preferredEmiDate: string
  disbursementAccount: DisbursementAccount
  emiStartPreference: EmiStartPreference
  prepaymentIntent: PrepaymentIntent
  insuranceOptIn: boolean

  // Section B · Financial position we cannot see
  obligationsOtherBanks: string
  additionalIncome: string
  additionalIncomeType: AdditionalIncomeType | ''
  expectedObligationsAfterLoan: string
  relationshipProducts: RelationshipProduct[]
  employmentChangedRecently: boolean

  // Section C · Security offered
  collateralType: CollateralType
  collateralDescription: string
  collateralValue: string
  collateralOwnershipProof: OwnershipProofType | ''
  collateralCoOwnerName: string

  // Section D · Guarantor
  guarantorName: string
  guarantorRelationship: GuarantorRelationship | ''
  guarantorContact: string
  guarantorBanksHere: boolean
  guarantorCustomerId: string
  guarantorConsentAck: boolean

  // Section E · Declarations
  endUseDeclaration: boolean
  isRelatedParty: boolean
  noWilfulDefault: boolean
  bureauPullConsent: boolean
  termsAccepted: boolean
}

const initialState: FormState = {
  customerId: '',

  loanAmount: '',
  loanPurpose: '',
  loanPurposeDetails: '',
  loanTenureMonths: '36',
  repaymentSource: 'salary',
  preferredEmiDate: '5',
  disbursementAccount: 'primary_savings',
  emiStartPreference: 'next_cycle',
  prepaymentIntent: 'none',
  insuranceOptIn: false,

  obligationsOtherBanks: '',
  additionalIncome: '',
  additionalIncomeType: '',
  expectedObligationsAfterLoan: '',
  relationshipProducts: [],
  employmentChangedRecently: false,

  collateralType: 'none',
  collateralDescription: '',
  collateralValue: '',
  collateralOwnershipProof: '',
  collateralCoOwnerName: '',

  guarantorName: '',
  guarantorRelationship: '',
  guarantorContact: '',
  guarantorBanksHere: false,
  guarantorCustomerId: '',
  guarantorConsentAck: false,

  endUseDeclaration: false,
  isRelatedParty: false,
  noWilfulDefault: false,
  bureauPullConsent: false,
  termsAccepted: false,
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

// Shared density vocabulary — same colours as before, tighter geometry.
const SECTION_CLASS = 'rounded-2xl border border-white/70 bg-white/85 p-5 shadow-lg backdrop-blur-xl'
const GRID_CLASS = 'grid grid-cols-1 gap-x-4 gap-y-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4'

export const CustomerNewApplication: React.FC = () => {
  const navigate = useNavigate()
  const { user } = useAuth()
  const { addApplication, isLoading, error } = useApplicationData({ scope: 'customer', applicantId: user?.uid })

  const [form, setForm] = useState<FormState>(initialState)
  const [profile, setProfile] = useState<CustomerProfile | null>(null)
  const [profileState, setProfileState] = useState<'idle' | 'loading' | 'found' | 'missing' | 'error'>('idle')
  const [submitError, setSubmitError] = useState<string | null>(null)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [result, setResult] = useState<LoanApplication | null>(null)

  // --- Sample customer ids (optional endpoint) ---------------------------
  const [samples, setSamples] = useState<CustomerSample[]>([])

  // --- Voice (optional module) ------------------------------------------
  const [voice, setVoice] = useState<VoiceStatus | null>(null)
  const [isRecording, setIsRecording] = useState(false)
  const [isTranscribing, setIsTranscribing] = useState(false)
  const [voiceNotice, setVoiceNotice] = useState<string | null>(null)
  const [isSpeaking, setIsSpeaking] = useState(false)
  const recorderRef = useRef<MediaRecorder | null>(null)
  const audioRef = useRef<HTMLAudioElement | null>(null)

  const setField = useCallback(<K extends keyof FormState>(key: K, value: FormState[K]) => {
    setForm((current) => ({ ...current, [key]: value }))
  }, [])

  // Sample ids are a convenience only. `getCustomerSamples` already swallows
  // every failure and resolves to [], so an absent endpoint simply means no
  // chips render. Nothing here can block or break the form.
  useEffect(() => {
    let cancelled = false
    getCustomerSamples()
      .then((rows) => {
        if (!cancelled) setSamples(rows)
      })
      .catch(() => {
        // Defensive belt-and-braces: never let this reject into the console.
        if (!cancelled) setSamples([])
      })
    return () => {
      cancelled = true
    }
  }, [])

  // Probe the voice module once on mount. `getVoiceStatus` resolves to null on
  // any failure, in which case no voice control is offered at all.
  useEffect(() => {
    let cancelled = false
    getVoiceStatus()
      .then((status) => {
        if (!cancelled) setVoice(status)
      })
      .catch(() => {
        if (!cancelled) setVoice(null)
      })
    return () => {
      cancelled = true
    }
  }, [])

  // Stop any in-flight recording / playback if the page unmounts mid-way.
  useEffect(() => {
    return () => {
      try {
        recorderRef.current?.stream.getTracks().forEach((track) => track.stop())
      } catch {
        /* nothing to clean up */
      }
      audioRef.current?.pause()
    }
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
        .then((profileResult) => {
          if (cancelled) return
          setProfile(profileResult)
          setProfileState(profileResult ? 'found' : 'missing')
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
  const hasCollateral = form.collateralType !== 'none'
  const guarantorRequired = !hasCollateral && loanAmount > GUARANTOR_THRESHOLD
  const hasAdditionalIncome = toNumber(form.additionalIncome) > 0

  const declarationsComplete =
    form.endUseDeclaration && form.noWilfulDefault && form.bureauPullConsent && form.termsAccepted

  const canSubmit =
    profileState === 'found'
    && loanAmount > 0
    && form.loanPurpose !== ''
    && declarationsComplete
    && (!hasCollateral || toNumber(form.collateralValue) > 0)
    && (!guarantorRequired
      || (form.guarantorRelationship !== '' && form.guarantorName.trim() !== '' && form.guarantorConsentAck))

  const toggleRelationshipProduct = (product: RelationshipProduct) => {
    setForm((current) => ({
      ...current,
      relationshipProducts: current.relationshipProducts.includes(product)
        ? current.relationshipProducts.filter((item) => item !== product)
        : [...current.relationshipProducts, product],
    }))
  }

  // ---- Voice: dictate into the purpose-details field ---------------------
  const startRecording = async () => {
    setVoiceNotice(null)
    // Guarded twice over: the button is only rendered when STT is configured
    // and MediaRecorder exists, but a permission denial can still happen here.
    if (!isRecordingSupported()) {
      setVoiceNotice('This browser cannot record audio.')
      return
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const recorder = new MediaRecorder(stream)
      const chunks: Blob[] = []
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunks.push(event.data)
      }
      recorder.onstop = () => {
        stream.getTracks().forEach((track) => track.stop())
        const blob = new Blob(chunks, { type: recorder.mimeType || 'audio/webm' })
        setIsRecording(false)
        if (blob.size === 0) return
        setIsTranscribing(true)
        transcribeAudio(blob)
          .then((transcription) => {
            if (!transcription) {
              // Graceful degradation: transcription unavailable or empty. Say
              // so quietly; the typed field is untouched and still usable.
              setVoiceNotice('Could not transcribe that — please type instead.')
              return
            }
            setForm((current) => ({
              ...current,
              loanPurposeDetails: current.loanPurposeDetails
                ? `${current.loanPurposeDetails.trim()} ${transcription.text}`
                : transcription.text,
            }))
          })
          .finally(() => setIsTranscribing(false))
      }
      recorderRef.current = recorder
      recorder.start()
      setIsRecording(true)
    } catch {
      // Mic permission denied or no input device — degrade to typing.
      setIsRecording(false)
      setVoiceNotice('Microphone unavailable — please type instead.')
    }
  }

  const stopRecording = () => {
    try {
      recorderRef.current?.stop()
    } catch {
      setIsRecording(false)
    }
  }

  // ---- Voice: read the decision aloud ------------------------------------
  const playDecision = async () => {
    if (!result) return
    setIsSpeaking(true)
    const decision = result.finalDecision ?? result.status ?? 'submitted'
    const explanation =
      result.decision?.explanation
      ?? `Your application for ${formatCurrency(result.loanAmount)} has been recorded and scored.`
    const audio = await synthesizeDecision(String(decision), explanation)
    if (!audio) {
      // Graceful degradation: no TTS key, or the provider call failed. The
      // written decision above stays exactly as it is.
      setIsSpeaking(false)
      setVoiceNotice('Audio playback is unavailable right now.')
      return
    }
    const url = URL.createObjectURL(audio)
    const element = new Audio(url)
    audioRef.current = element
    element.onended = () => {
      setIsSpeaking(false)
      URL.revokeObjectURL(url)
    }
    element.onerror = () => {
      setIsSpeaking(false)
      URL.revokeObjectURL(url)
    }
    try {
      await element.play()
    } catch {
      // Autoplay blocked — silent no-op, voice is never load-bearing.
      setIsSpeaking(false)
      URL.revokeObjectURL(url)
    }
  }

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
      loan_purpose_details: form.loanPurposeDetails.trim() || undefined,
      loan_tenure_months: toNumber(form.loanTenureMonths),
      repayment_source: form.repaymentSource,
      preferred_emi_date: toNumber(form.preferredEmiDate),
      disbursement_account: form.disbursementAccount,
      emi_start_preference: form.emiStartPreference,
      prepayment_intent: form.prepaymentIntent,
      insurance_opt_in: form.insuranceOptIn,

      obligations_other_banks: toNumber(form.obligationsOtherBanks),
      employment_changed_recently: form.employmentChangedRecently,
      additional_income: toNumber(form.additionalIncome),
      ...(hasAdditionalIncome && form.additionalIncomeType
        ? { additional_income_type: form.additionalIncomeType }
        : {}),
      expected_obligations_after_loan: toNumber(form.expectedObligationsAfterLoan),
      relationship_products: form.relationshipProducts,

      collateral_type: form.collateralType,
      collateral_value: hasCollateral ? toNumber(form.collateralValue) : 0,
      ...(hasCollateral
        ? {
            collateral_description: form.collateralDescription.trim() || undefined,
            collateral_ownership_proof: (form.collateralOwnershipProof || undefined) as
              | OwnershipProofType
              | undefined,
            collateral_co_owner_name: form.collateralCoOwnerName.trim() || undefined,
          }
        : {}),

      ...(guarantorRequired
        ? {
            guarantor_name: form.guarantorName.trim() || undefined,
            guarantor_relationship: form.guarantorRelationship as GuarantorRelationship,
            guarantor_contact: form.guarantorContact.trim() || undefined,
            guarantor_banks_here: form.guarantorBanksHere,
            guarantor_customer_id: form.guarantorCustomerId.trim() || undefined,
            guarantor_consent_ack: form.guarantorConsentAck,
          }
        : {}),

      end_use_declaration: form.endUseDeclaration,
      is_related_party: form.isRelatedParty,
      no_wilful_default: form.noWilfulDefault,
      bureau_pull_consent: form.bureauPullConsent,
      terms_accepted: form.termsAccepted,
    }

    try {
      const application = await addApplication(payload)
      trackEvent('application_submitted', { applicationId: application.id })
      setResult(application)
    } catch (submitFailure) {
      setSubmitError(submitFailure instanceof Error ? submitFailure.message : 'Failed to submit application')
    } finally {
      setIsSubmitting(false)
    }
  }

  // Only offer the mic when STT is actually configured AND the browser can
  // record. Otherwise a small disabled hint is shown instead of a dead button.
  const canDictate = Boolean(voice?.stt.configured) && isRecordingSupported()
  const canListen = Boolean(voice?.tts.configured)

  return (
    <DashboardLayout title="Customer Dashboard" role="customer">
      <section className="mb-4">
        <Button
          variant="ghost"
          leftIcon={<ChevronLeft className="h-4 w-4" />}
          onClick={() => navigate('/dashboard/customer')}
        >
          Back to dashboard
        </Button>
      </section>

      {(error || submitError) && (
        <section className="mb-4">
          <Card className="border-red-200 bg-red-50">
            <p className="text-red-700">{submitError ?? error}</p>
          </Card>
        </section>
      )}

      {result ? (
        <DecisionResult
          application={result}
          canListen={canListen}
          isSpeaking={isSpeaking}
          voiceNotice={voiceNotice}
          onListen={() => void playDecision()}
          onDone={() => navigate('/dashboard/customer?view=history')}
        />
      ) : (
        <section>
          <Card
            title="New Loan Application"
            description="You are an existing customer, so we only ask for what we cannot already see. Your KYC, income, employment and bureau record are pulled automatically."
            className="rounded-2xl border-white/80"
          >
            <form onSubmit={handleSubmit} className="space-y-5">
              {/* ---- Customer identification + read-only pulled profile ---- */}
              <section className={SECTION_CLASS}>
                <SectionHeading
                  eyebrow="Your Account"
                  title="Customer Identification"
                  icon={<UserCheck className="h-5 w-5" />}
                  tone="primary"
                />

                <div className={GRID_CLASS}>
                  <Input
                    label="Customer ID"
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

                  {/* Sample-id chips. Rendered only when the optional
                      /api/customers/samples endpoint answered with rows; when
                      it 404s or errors, `samples` stays [] and this whole
                      block disappears without a trace. */}
                  {samples.length > 0 && (
                    <div className="sm:col-span-2 lg:col-span-3">
                      <p className="mb-2 text-sm font-medium text-neutral-700">Try a sample customer</p>
                      <div className="flex flex-wrap gap-1.5">
                        {samples.map((sample) => (
                          <button
                            key={sample.customer_id}
                            type="button"
                            onClick={() => setField('customerId', sample.customer_id)}
                            className={`rounded-full border px-3 py-1 text-xs font-medium transition-colors ${
                              form.customerId.trim() === sample.customer_id
                                ? 'border-primary-500 bg-primary-100 text-primary-900'
                                : 'border-neutral-200 bg-white text-neutral-500 hover:border-primary-500 hover:bg-primary-50 hover:text-primary-900'
                            }`}
                          >
                            <span className="font-semibold">{sample.customer_id}</span>
                            {sample.descriptor && <span className="ml-1.5 opacity-70">{sample.descriptor}</span>}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                {profileState === 'loading' && (
                  <p className="mt-3 text-sm text-neutral-500">Looking up your record…</p>
                )}

                {profileState === 'found' && profile && (
                  <div className="mt-4 rounded-xl border border-primary-100 bg-primary-50/60 p-4">
                    <div className="flex items-center gap-2">
                      <ShieldCheck className="h-4 w-4 text-primary-600" />
                      <p className="text-sm font-semibold text-primary-900">
                        We already have this — no need to re-enter it
                      </p>
                    </div>
                    <p className="mt-1 text-xs text-primary-800/80">
                      Read-only. Pulled from your KYC record, your account history and your bureau report.
                    </p>

                    <div className="mt-3 grid grid-cols-1 gap-3 md:grid-cols-3">
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
              <section className={SECTION_CLASS}>
                <SectionHeading
                  eyebrow="Section A"
                  title="Loan Request"
                  subtitle="Amount, structure and how you want the money disbursed and repaid."
                  icon={<Wallet className="h-5 w-5" />}
                  tone="accent"
                />

                <div className={GRID_CLASS}>
                  <Input
                    label="Loan amount (₹)"
                    type="number"
                    min={0}
                    required
                    value={form.loanAmount}
                    onChange={(event) => setField('loanAmount', event.target.value)}
                  />

                  <Select
                    label="Loan purpose"
                    required
                    options={LOAN_PURPOSE_OPTIONS}
                    value={form.loanPurpose}
                    onChange={(value) => setField('loanPurpose', String(value) as LoanPurposeShort)}
                  />

                  <Select
                    label="Tenure"
                    required
                    options={TENURE_OPTIONS}
                    value={form.loanTenureMonths}
                    onChange={(value) => setField('loanTenureMonths', String(value))}
                  />

                  <Select
                    label="Repayment source"
                    required
                    options={REPAYMENT_SOURCE_OPTIONS}
                    value={form.repaymentSource}
                    onChange={(value) => setField('repaymentSource', String(value) as RepaymentSource)}
                  />

                  <Select
                    label="Disbursement account"
                    options={DISBURSEMENT_ACCOUNT_OPTIONS}
                    value={form.disbursementAccount}
                    onChange={(value) => setField('disbursementAccount', String(value) as DisbursementAccount)}
                  />

                  <Select
                    label="First EMI should start"
                    options={EMI_START_OPTIONS}
                    value={form.emiStartPreference}
                    onChange={(value) => setField('emiStartPreference', String(value) as EmiStartPreference)}
                  />

                  <Select
                    label="Preferred EMI date"
                    options={EMI_DATE_OPTIONS}
                    value={form.preferredEmiDate}
                    onChange={(value) => setField('preferredEmiDate', String(value))}
                  />

                  <Select
                    label="Prepayment intent"
                    options={PREPAYMENT_OPTIONS}
                    value={form.prepaymentIntent}
                    onChange={(value) => setField('prepaymentIntent', String(value) as PrepaymentIntent)}
                  />
                </div>

                <div className="mt-3 grid grid-cols-1 gap-3 lg:grid-cols-3">
                  <div className="lg:col-span-2">
                    <div className="mb-2 flex items-center justify-between gap-2">
                      <label className="text-sm font-medium text-neutral-700" htmlFor="loan-purpose-details">
                        What exactly is the money for?
                      </label>

                      {/* Voice dictation. Three distinct states, all honest:
                          (1) STT configured + browser can record → live mic,
                          (2) voice module reachable but unconfigured → small
                              disabled hint naming the missing key,
                          (3) status probe failed entirely → nothing at all. */}
                      {canDictate ? (
                        <button
                          type="button"
                          onClick={() => (isRecording ? stopRecording() : void startRecording())}
                          disabled={isTranscribing}
                          className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium transition-colors disabled:opacity-60 ${
                            isRecording
                              ? 'bg-red-100 text-red-700 hover:bg-red-200'
                              : 'bg-primary-50 text-primary-700 hover:bg-primary-100'
                          }`}
                        >
                          {isRecording ? <Square className="h-3 w-3 fill-current" /> : <Mic className="h-3 w-3" />}
                          {isRecording ? 'Recording — tap to stop' : isTranscribing ? 'Transcribing…' : 'Dictate'}
                        </button>
                      ) : voice ? (
                        <span
                          className="inline-flex cursor-not-allowed items-center gap-1.5 rounded-full bg-neutral-100 px-3 py-1 text-xs font-medium text-neutral-400"
                          title={voice.stt.reason ?? 'Speech-to-text is not configured on this deployment.'}
                        >
                          <Mic className="h-3 w-3" />
                          Dictation off — API key needed
                        </span>
                      ) : null}
                    </div>

                    <Textarea
                      id="loan-purpose-details"
                      rows={3}
                      value={form.loanPurposeDetails}
                      onChange={(event) => setField('loanPurposeDetails', event.target.value)}
                      placeholder="e.g. Rewiring and waterproofing the first floor of my house."
                      hint={voiceNotice ?? 'Optional. Helps the reviewer understand the end use.'}
                    />
                  </div>

                  <CheckboxCard
                    checked={form.insuranceOptIn}
                    onChange={(checked) => setField('insuranceOptIn', checked)}
                    title="Loan protection insurance"
                    body="Add optional cover; the premium is financed with the loan."
                  />
                </div>
              </section>

              {/* ---- Section B · Financial position we cannot see ---- */}
              <section className={SECTION_CLASS}>
                <SectionHeading
                  eyebrow="Section B"
                  title="Financial Position We Cannot See"
                  subtitle="We see our own accounts and your bureau report, but bureau data lags 30–45 days."
                  icon={<Landmark className="h-5 w-5" />}
                  tone="secondary"
                />

                <div className={GRID_CLASS}>
                  <Input
                    label="Monthly EMIs at other lenders (₹)"
                    type="number"
                    min={0}
                    value={form.obligationsOtherBanks}
                    onChange={(event) => setField('obligationsOtherBanks', event.target.value)}
                    hint="Obligations we cannot observe"
                  />

                  <Input
                    label="Additional monthly income (₹)"
                    type="number"
                    min={0}
                    value={form.additionalIncome}
                    onChange={(event) => setField('additionalIncome', event.target.value)}
                    hint="Rental, freelance, spouse income"
                  />

                  {/* Conditional reveal: the type only matters once an amount
                      has been declared. */}
                  {hasAdditionalIncome && (
                    <Select
                      label="Source of that income"
                      options={ADDITIONAL_INCOME_TYPE_OPTIONS}
                      value={form.additionalIncomeType}
                      onChange={(value) =>
                        setField('additionalIncomeType', String(value) as AdditionalIncomeType)
                      }
                    />
                  )}

                  <Input
                    label="Expected monthly obligations after this loan (₹)"
                    type="number"
                    min={0}
                    value={form.expectedObligationsAfterLoan}
                    onChange={(event) => setField('expectedObligationsAfterLoan', event.target.value)}
                    hint="Including the new EMI and any planned commitments"
                  />
                </div>

                <div className="mt-4">
                  <p className="mb-2 text-sm font-medium text-neutral-700">
                    Products you already hold with us
                  </p>
                  <div className="flex flex-wrap gap-1.5">
                    {RELATIONSHIP_PRODUCT_OPTIONS.map((option) => {
                      const selected = form.relationshipProducts.includes(option.value)
                      return (
                        <button
                          key={option.value}
                          type="button"
                          onClick={() => toggleRelationshipProduct(option.value)}
                          className={`rounded-full border px-3 py-1 text-xs font-medium transition-colors ${
                            selected
                              ? 'border-primary-500 bg-primary-50 text-primary-900'
                              : 'border-neutral-200 bg-white text-neutral-500 hover:bg-neutral-50'
                          }`}
                        >
                          {option.label}
                        </button>
                      )
                    })}
                  </div>
                </div>

                <div className="mt-3">
                  <CheckboxCard
                    checked={form.employmentChangedRecently}
                    onChange={(checked) => setField('employmentChangedRecently', checked)}
                    title="Job change in the last 6 months"
                    body="A recent change postdates your last KYC update and materially changes risk."
                  />
                </div>
              </section>

              {/* ---- Section C · Security offered ---- */}
              <section className={SECTION_CLASS}>
                <SectionHeading
                  eyebrow="Section C"
                  title="Security Offered"
                  subtitle="Pledging security can remove the guarantor requirement on larger loans."
                  icon={<Gem className="h-5 w-5" />}
                  tone="accent"
                />

                <div className={GRID_CLASS}>
                  <Select
                    label="Collateral offered"
                    options={COLLATERAL_OPTIONS}
                    value={form.collateralType}
                    onChange={(value) => setField('collateralType', String(value) as CollateralType)}
                  />

                  {/* Conditional reveal: nothing below is meaningful unsecured. */}
                  {hasCollateral && (
                    <>
                      <Input
                        label="Estimated value (₹)"
                        type="number"
                        min={0}
                        required
                        value={form.collateralValue}
                        onChange={(event) => setField('collateralValue', event.target.value)}
                      />

                      <Select
                        label="Ownership proof type"
                        options={OWNERSHIP_PROOF_OPTIONS}
                        value={form.collateralOwnershipProof}
                        onChange={(value) =>
                          setField('collateralOwnershipProof', String(value) as OwnershipProofType)
                        }
                      />

                      <Input
                        label="Co-owner name"
                        value={form.collateralCoOwnerName}
                        onChange={(event) => setField('collateralCoOwnerName', event.target.value)}
                        hint="Leave blank if solely owned"
                      />

                      <div className="sm:col-span-2 lg:col-span-3 xl:col-span-4">
                        <Input
                          label="Description of the security"
                          value={form.collateralDescription}
                          onChange={(event) => setField('collateralDescription', event.target.value)}
                          placeholder="e.g. 2BHK flat, 780 sq ft, Survey no. 114/2, Pune"
                          hint="Enough detail for the valuer to identify it"
                        />
                      </div>
                    </>
                  )}
                </div>
              </section>

              {/* ---- Section D · Guarantor (conditional annexure) ---- */}
              {guarantorRequired && (
                <section className={SECTION_CLASS}>
                  <SectionHeading
                    eyebrow="Section D"
                    title="Guarantor"
                    subtitle={`Required because this request is unsecured and above ₹${GUARANTOR_THRESHOLD.toLocaleString()}.`}
                    icon={<Users className="h-5 w-5" />}
                    tone="primary"
                  />

                  <div className={GRID_CLASS}>
                    <Input
                      label="Guarantor full name"
                      required
                      value={form.guarantorName}
                      onChange={(event) => setField('guarantorName', event.target.value)}
                    />

                    <Select
                      label="Relationship to you"
                      required
                      options={GUARANTOR_RELATIONSHIP_OPTIONS}
                      value={form.guarantorRelationship}
                      onChange={(value) =>
                        setField('guarantorRelationship', String(value) as GuarantorRelationship)
                      }
                    />

                    <Input
                      label="Guarantor contact number"
                      type="tel"
                      value={form.guarantorContact}
                      onChange={(event) => setField('guarantorContact', event.target.value)}
                      hint="We will call to confirm consent"
                    />

                    {/* Conditional reveal: only ask for the id once they say
                        the guarantor banks with us. */}
                    {form.guarantorBanksHere && (
                      <Input
                        label="Guarantor customer ID"
                        value={form.guarantorCustomerId}
                        onChange={(event) => setField('guarantorCustomerId', event.target.value)}
                        hint="We pull their profile instead of re-keying it"
                      />
                    )}
                  </div>

                  <div className="mt-3 grid grid-cols-1 gap-3 lg:grid-cols-2">
                    <CheckboxCard
                      checked={form.guarantorBanksHere}
                      onChange={(checked) => setField('guarantorBanksHere', checked)}
                      title="The guarantor also banks with us"
                      body="Lets us pull their record rather than collecting it again."
                    />
                    <CheckboxCard
                      checked={form.guarantorConsentAck}
                      onChange={(checked) => setField('guarantorConsentAck', checked)}
                      title="Guarantor consent acknowledgement (required)"
                      body="I confirm the named guarantor has agreed to stand surety and to a bureau enquiry."
                    />
                  </div>
                </section>
              )}

              {/* ---- Section E · Declarations ---- */}
              <section className={SECTION_CLASS}>
                <SectionHeading
                  eyebrow="Section E"
                  title="Declarations & Consents"
                  subtitle="Statutory declarations every Indian lender is required to collect."
                  icon={<FileText className="h-5 w-5" />}
                  tone="secondary"
                />

                <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
                  <CheckboxCard
                    checked={form.endUseDeclaration}
                    onChange={(checked) => setField('endUseDeclaration', checked)}
                    title="End-use declaration (required)"
                    body="I confirm the funds will not be used for speculative or anti-social purposes."
                  />
                  <CheckboxCard
                    checked={form.noWilfulDefault}
                    onChange={(checked) => setField('noWilfulDefault', checked)}
                    title="Wilful-defaulter declaration (required)"
                    body="I am not classified as a wilful defaulter by any bank or financial institution."
                  />
                  <CheckboxCard
                    checked={form.bureauPullConsent}
                    onChange={(checked) => setField('bureauPullConsent', checked)}
                    title="Credit-bureau enquiry consent (required)"
                    body="I authorise the bank to obtain my credit information report from CIBIL / Experian."
                  />
                  <CheckboxCard
                    checked={form.termsAccepted}
                    onChange={(checked) => setField('termsAccepted', checked)}
                    title="Terms and conditions (required)"
                    body="I have read the sanction terms and confirm the information given is true and complete."
                  />
                  <CheckboxCard
                    checked={form.isRelatedParty}
                    onChange={(checked) => setField('isRelatedParty', checked)}
                    title="Related-party declaration (RBI)"
                    body="I am a director of a bank, or a relative of one."
                  />
                </div>
              </section>

              {profileState !== 'found' && (
                <div className="rounded-lg border-l-4 border-blue-500 bg-blue-50 p-4">
                  <div className="flex gap-3">
                    <AlertCircle className="mt-0.5 h-5 w-5 flex-shrink-0 text-blue-600" />
                    <div>
                      <p className="font-medium text-blue-900">Enter your customer ID to continue</p>
                      <p className="mt-1 text-sm text-blue-800">
                        We score the application against the record we already hold on you, so the ID must
                        resolve before it can be submitted.
                        {samples.length > 0 && ' Pick one of the sample IDs above to try it out.'}
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
                className="rounded-xl py-3 shadow-lg"
              >
                {isSubmitting || isLoading ? 'Submitting...' : 'Submit Application'}
              </Button>
            </form>
          </Card>
        </section>
      )}
    </DashboardLayout>
  )
}

function SectionHeading({
  eyebrow,
  title,
  subtitle,
  icon,
  tone,
}: {
  eyebrow: string
  title: string
  subtitle?: string
  icon: React.ReactNode
  tone: 'primary' | 'accent' | 'secondary'
}) {
  // Only tokens that exist in tailwind.config.js — the palette defines
  // primary/accent/neutral and flat semantic colours, but no `secondary` scale.
  const toneClass =
    tone === 'primary'
      ? 'bg-primary-50 text-primary-600'
      : tone === 'accent'
        ? 'bg-accent-50 text-accent-500'
        : 'bg-primary-100 text-primary-900'

  return (
    <div className="mb-4 flex items-start justify-between gap-4">
      <div>
        <p className="text-[11px] uppercase tracking-[0.22em] text-neutral-500">{eyebrow}</p>
        <h3 className="mt-1 text-lg font-semibold text-neutral-900">{title}</h3>
        {subtitle && <p className="mt-1 max-w-3xl text-sm text-neutral-500">{subtitle}</p>}
      </div>
      <div className={`shrink-0 rounded-xl p-2.5 ${toneClass}`}>{icon}</div>
    </div>
  )
}

function CheckboxCard({
  checked,
  onChange,
  title,
  body,
}: {
  checked: boolean
  onChange: (checked: boolean) => void
  title: string
  body: string
}) {
  return (
    <label className="flex cursor-pointer items-start gap-2.5 rounded-xl border border-neutral-200 bg-neutral-50 p-3 transition-colors hover:border-neutral-300 hover:bg-white">
      <input
        type="checkbox"
        className="mt-0.5 h-4 w-4 rounded border-neutral-300"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
      />
      <span className="text-sm text-neutral-700">
        <span className="block font-medium text-neutral-900">{title}</span>
        <span className="mt-0.5 block text-xs text-neutral-500">{body}</span>
      </span>
    </label>
  )
}

function DecisionResult({
  application,
  canListen,
  isSpeaking,
  voiceNotice,
  onListen,
  onDone,
}: {
  application: LoanApplication
  canListen: boolean
  isSpeaking: boolean
  voiceNotice: string | null
  onListen: () => void
  onDone: () => void
}) {
  const decision = (application.finalDecision ?? application.status ?? 'submitted').toString()
  const explanation = application.decision?.explanation

  return (
    <section>
      <Card title="Application Submitted" className="rounded-2xl border-white/80">
        <div className="space-y-4">
          <div className="flex items-start gap-3 rounded-xl border border-primary-100 bg-primary-50/60 p-4">
            <CheckCircle2 className="mt-0.5 h-5 w-5 shrink-0 text-primary-600" />
            <div>
              <p className="text-sm font-semibold text-primary-900">
                Decision: <span className="uppercase tracking-wide">{decision}</span>
              </p>
              <p className="mt-1 text-sm text-primary-800/80">
                {explanation ?? 'Your application has been recorded and scored. The full breakdown is in your history.'}
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
            <ProfileBlock
              title="Request"
              rows={[
                ['Amount', formatCurrency(application.loanAmount)],
                ['Tenure', `${application.loanTenure ?? 0} months`],
                ['Reference', application.id],
              ]}
            />
            <ProfileBlock
              title="Model output"
              rows={[
                ['ML probability', formatPercent(application.ml_prob)],
                ['CBES probability', formatPercent(application.cbes_prob)],
                ['Confidence', formatPercent(application.confidence)],
              ]}
            />
            <ProfileBlock
              title="Status"
              rows={[
                ['Status', formatText(application.status)],
                ['Recommendation', formatText(application.modelRecommendation)],
                ['Submitted', new Date(application.createdAt ?? Date.now()).toLocaleString()],
              ]}
            />
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <Button variant="primary" onClick={onDone}>
              View my applications
            </Button>

            {/* Graceful degradation: the listen button only exists when
                /api/voice/status reported a configured TTS provider. With no
                key the decision above is simply text-only — nothing broken,
                nothing disabled-and-confusing. */}
            {canListen && (
              <Button
                variant="secondary"
                leftIcon={<Volume2 className="h-4 w-4" />}
                onClick={onListen}
                isLoading={isSpeaking}
              >
                {isSpeaking ? 'Playing…' : 'Listen to decision'}
              </Button>
            )}
          </div>

          {voiceNotice && <p className="text-xs text-neutral-500">{voiceNotice}</p>}
        </div>
      </Card>
    </section>
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
    <div className="rounded-xl border border-neutral-200 bg-white p-3">
      <p className="text-[11px] uppercase tracking-[0.18em] text-neutral-500">{title}</p>
      <div className="mt-2 space-y-1.5 text-sm text-neutral-700">
        {rows.map(([label, value]) => (
          <p key={label}>
            <span className="font-medium text-neutral-900">{label}:</span> {value}
          </p>
        ))}
      </div>
      {footnote && <p className="mt-2 text-xs text-neutral-400">{footnote}</p>}
    </div>
  )
}
