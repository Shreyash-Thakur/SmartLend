import React, { useEffect, useMemo, useState } from 'react'
import { Controller } from 'react-hook-form'
import { AlertCircle, FileText, Landmark, MapPin, Wallet } from 'lucide-react'
import { Button, Input, Select } from '@/components/common'
import { FileUploadArea } from '@/components/forms'
import type { LoanApplicationFormData } from '@/types/application'
import type { ApplicationFormProps } from '@/types/ui'
import { loanApplicationSchema, useFormValidation } from '@/hooks/useFormValidation'

const INTEREST_RATE_SLABS = {
  personal: { min: 10.5, max: 14.0, default: 12.0 },
  home: { min: 8.0, max: 9.0, default: 8.5 },
  business: { min: 12.0, max: 15.0, default: 13.5 },
  auto: { min: 9.0, max: 11.0, default: 10.0 },
  education: { min: 7.0, max: 8.5, default: 7.75 },
}

const MEAN_CREDIT_UTILIZATION_RATIO = 0.35

function displayValue(value: string | number | undefined | null) {
  if (value === undefined || value === null || value === '') return 'N/A'
  return String(value)
}

const defaultValues: Partial<LoanApplicationFormData> = {
  firstName: '',
  lastName: '',
  email: '',
  phone: '',
  gender: '' as LoanApplicationFormData['gender'],
  maritalStatus: '' as LoanApplicationFormData['maritalStatus'],
  age: undefined,
  dependents: undefined,
  employmentType: '' as LoanApplicationFormData['employmentType'],
  yearsOfEmployment: undefined,
  monthlyIncome: undefined,
  annualIncome: undefined,
  loanPurpose: '' as LoanApplicationFormData['loanPurpose'],
  loanAmount: undefined,
  loanTenure: 36,
  interestRate: undefined,
  emi: 0,
  existingEmis: undefined,
  residentialAssetsValue: undefined,
  commercialAssetsValue: undefined,
  bankBalance: undefined,
  totalAssets: undefined,
  assets: undefined,
  liabilities: undefined,
  cibilScore: undefined,
  totalLoans: undefined,
  activeLoans: undefined,
  closedLoans: undefined,
  missedPayments: undefined,
  creditUtilizationRatio: MEAN_CREDIT_UTILIZATION_RATIO,
  emiIncomeRatio: undefined,
  loanIncomeRatio: undefined,
  debtToIncomeRatio: undefined,
  region: '' as LoanApplicationFormData['region'],
  city: '',
}

const steps = [
  'Identity',
  'Income & Loan',
  'Assets & Credit',
  'Documents',
  'Review',
]

export const LoanApplicationForm: React.FC<ApplicationFormProps> = ({
  onSubmit,
  initialData,
  isLoading = false,
  isMultiStep = true,
}) => {
  const storageKey = 'smartlend.loan-application.draft.v3'
  const savedDraft = useMemo(() => {
    const raw = localStorage.getItem(storageKey)
    return raw ? (JSON.parse(raw) as Partial<LoanApplicationFormData>) : undefined
  }, [])
  const [currentStep, setCurrentStep] = useState(0)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)

  const {
    control,
    register,
    handleSubmit,
    watch,
    formState: { errors, isSubmitting, isValid },
    reset,
    setValue,
  } = useFormValidation(loanApplicationSchema)

  const values = watch()
  const monthlyIncome = values.monthlyIncome
  const annualIncome = monthlyIncome ? monthlyIncome * 12 : values.annualIncome
  const totalAssets =
    (values.residentialAssetsValue || 0) +
    (values.commercialAssetsValue || 0) +
    (values.bankBalance || 0)
  const totalEmis = (values.emi || 0) + (values.existingEmis || 0)
  const emiIncomeRatio = monthlyIncome ? (totalEmis / monthlyIncome) * 100 : undefined
  const requestedLoanAmount = values.loanAmount
  const loanIncomeRatio = annualIncome && requestedLoanAmount ? (requestedLoanAmount / annualIncome) * 100 : undefined
  const debtToIncomeRatio =
    annualIncome ? (((values.liabilities || 0) + totalEmis * 12) / annualIncome) * 100 : undefined

  const getCreditCategory = (score: number | undefined) => {
    if (!score) return 'N/A'
    if (score > 750) return 'Good'
    if (score >= 600) return 'Average'
    return 'Bad'
  }

  const getInterestRate = (loanPurpose: keyof typeof INTEREST_RATE_SLABS | '', income: number) => {
    if (!loanPurpose || !INTEREST_RATE_SLABS[loanPurpose]) return 0
    const slab = INTEREST_RATE_SLABS[loanPurpose]
    if (income >= 150000) return slab.min
    if (income >= 75000) return Number(((slab.min + slab.max) / 2).toFixed(2))
    return slab.max
  }

  useEffect(() => {
    reset({
      ...defaultValues,
      ...savedDraft,
      ...initialData,
    })
  }, [initialData, reset, savedDraft])

  useEffect(() => {
    if (monthlyIncome) {
      setValue('annualIncome', Math.round(monthlyIncome * 12))
    } else {
      setValue('annualIncome', undefined)
    }

    const hasAssetInput =
      values.residentialAssetsValue !== undefined
      || values.commercialAssetsValue !== undefined
      || values.bankBalance !== undefined
    const totalAssetsValue = hasAssetInput ? Math.round(totalAssets) : undefined

    setValue('totalAssets', totalAssetsValue)
    setValue('assets', Math.round(totalAssets))
    setValue('emiIncomeRatio', emiIncomeRatio === undefined ? undefined : Number(emiIncomeRatio.toFixed(2)))
    setValue('loanIncomeRatio', loanIncomeRatio === undefined ? undefined : Number(loanIncomeRatio.toFixed(2)))
    setValue('debtToIncomeRatio', debtToIncomeRatio === undefined ? undefined : Number(debtToIncomeRatio.toFixed(2)))
    setValue('creditUtilizationRatio', MEAN_CREDIT_UTILIZATION_RATIO)
  }, [
    annualIncome,
    debtToIncomeRatio,
    emiIncomeRatio,
    loanIncomeRatio,
    monthlyIncome,
    setValue,
    totalAssets,
    values.bankBalance,
    values.commercialAssetsValue,
    values.residentialAssetsValue,
  ])

  useEffect(() => {
    const loanPurpose = values.loanPurpose as keyof typeof INTEREST_RATE_SLABS
    if (loanPurpose && INTEREST_RATE_SLABS[loanPurpose]) {
      setValue('interestRate', getInterestRate(loanPurpose, monthlyIncome))
    }
  }, [monthlyIncome, values.loanPurpose, setValue])

  useEffect(() => {
    const subscription = watch((value) => {
      localStorage.setItem(storageKey, JSON.stringify(value))
    })
    return () => subscription.unsubscribe()
  }, [watch])

  const submitForm = async (data: LoanApplicationFormData) => {
    await onSubmit(data)
    localStorage.removeItem(storageKey)
    if (isMultiStep) {
      setCurrentStep(0)
    }
  }

  const renderStepVisibility = (index: number) => !isMultiStep || currentStep === index

  return (
    <form onSubmit={handleSubmit(submitForm)} className="space-y-8">
      {isMultiStep && (
        <div className="rounded-[28px] border border-white/70 bg-white/80 p-5 shadow-lg backdrop-blur-xl">
          <div className="grid gap-3 md:grid-cols-4">
            {steps.map((step, index) => (
              <button
                key={step}
                type="button"
                onClick={() => setCurrentStep(index)}
                className={`rounded-2xl px-4 py-3 text-left transition-all ${
                  index === currentStep
                    ? 'bg-neutral-900 text-white shadow-md'
                    : index < currentStep
                      ? 'bg-primary-50 text-primary-900'
                      : 'bg-neutral-100 text-neutral-500'
                }`}
              >
                <div className="text-xs uppercase tracking-[0.2em] opacity-70">Step {index + 1}</div>
                <div className="mt-1 text-sm font-semibold">{step}</div>
              </button>
            ))}
          </div>
        </div>
      )}

      {renderStepVisibility(0) && (
        <section className="rounded-[32px] border border-white/70 bg-white/85 p-6 shadow-xl backdrop-blur-xl">
          <div className="mb-5 flex items-center justify-between">
            <div>
              <p className="text-xs uppercase tracking-[0.24em] text-neutral-500">Personal Information</p>
              <h3 className="mt-2 text-2xl font-semibold text-neutral-900">Identity and Contact</h3>
            </div>
            <div className="rounded-2xl bg-primary-50 p-3 text-primary-600">
              <MapPin className="h-6 w-6" />
            </div>
          </div>

          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
            <Input label="First Name *" required {...register('firstName')} error={errors.firstName?.message} />
            <Input label="Last Name *" required {...register('lastName')} error={errors.lastName?.message} />
            <Input label="Email *" type="email" required {...register('email')} error={errors.email?.message} />
            <Input label="Phone *" required {...register('phone')} error={errors.phone?.message} />
            <Input label="City" {...register('city')} error={errors.city?.message} />

            <Controller
              control={control}
              name="gender"
              render={({ field }) => (
                <Select
                  label="Gender *"
                  options={[
                    { value: '', label: 'Select...' },
                    { value: 'male', label: 'Male' },
                    { value: 'female', label: 'Female' },
                    { value: 'other', label: 'Other' },
                  ]}
                  value={field.value || ''}
                  onChange={field.onChange}
                  error={errors.gender?.message}
                />
              )}
            />

            <Controller
              control={control}
              name="maritalStatus"
              render={({ field }) => (
                <Select
                  label="Marital Status"
                  options={[
                    { value: '', label: 'Select...' },
                    { value: 'single', label: 'Single' },
                    { value: 'married', label: 'Married' },
                    { value: 'divorced', label: 'Divorced' },
                    { value: 'widowed', label: 'Widowed' },
                  ]}
                  value={field.value || ''}
                  onChange={field.onChange}
                  error={errors.maritalStatus?.message}
                />
              )}
            />

            <Input label="Age *" type="number" required {...register('age', { valueAsNumber: true })} error={errors.age?.message} />
            <Input
              label="Dependents"
              type="number"
              {...register('dependents', { valueAsNumber: true })}
              error={errors.dependents?.message}
            />

            <Controller
              control={control}
              name="region"
              render={({ field }) => (
                <Select
                  label="Region Type *"
                  options={[
                    { value: '', label: 'Select...' },
                    { value: 'rural', label: 'Rural' },
                    { value: 'urban', label: 'Urban' },
                    { value: 'semi_urban', label: 'Semi-Urban' },
                  ]}
                  value={field.value || ''}
                  onChange={field.onChange}
                  error={errors.region?.message}
                />
              )}
            />
          </div>
        </section>
      )}

      {renderStepVisibility(1) && (
        <section className="rounded-[32px] border border-white/70 bg-white/85 p-6 shadow-xl backdrop-blur-xl">
          <div className="mb-5 flex items-center justify-between">
            <div>
              <p className="text-xs uppercase tracking-[0.24em] text-neutral-500">Income & Employment</p>
              <h3 className="mt-2 text-2xl font-semibold text-neutral-900">Employment and Loan Request</h3>
            </div>
            <div className="rounded-2xl bg-accent-50 p-3 text-accent-500">
              <Wallet className="h-6 w-6" />
            </div>
          </div>

          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
            <Controller
              control={control}
              name="employmentType"
              render={({ field }) => (
                <Select
                  label="Employment Type *"
                  options={[
                    { value: '', label: 'Select...' },
                    { value: 'salaried', label: 'Salaried' },
                    { value: 'self-employed', label: 'Self-Employed' },
                    { value: 'business', label: 'Business' },
                    { value: 'retired', label: 'Retired' },
                  ]}
                  value={field.value || ''}
                  onChange={field.onChange}
                  error={errors.employmentType?.message}
                />
              )}
            />

            <Input
              label="Years of Employment"
              type="number"
              {...register('yearsOfEmployment', { valueAsNumber: true })}
              error={errors.yearsOfEmployment?.message}
            />

            <Input label="Monthly Income *" type="number" required {...register('monthlyIncome', { valueAsNumber: true })} error={errors.monthlyIncome?.message} />

            <Input
              label="Annual Income (Auto)"
              type="number"
              disabled
              {...register('annualIncome', { valueAsNumber: true })}
            />

            <Controller
              control={control}
              name="loanPurpose"
              render={({ field }) => (
                <Select
                  label="Loan Purpose *"
                  options={[
                    { value: '', label: 'Select...' },
                    { value: 'personal', label: 'Personal' },
                    { value: 'home', label: 'Home' },
                    { value: 'business', label: 'Business' },
                    { value: 'auto', label: 'Auto' },
                    { value: 'education', label: 'Education' },
                  ]}
                  value={field.value || ''}
                  onChange={field.onChange}
                  error={errors.loanPurpose?.message}
                />
              )}
            />

            <Input label="Loan Amount *" type="number" required {...register('loanAmount', { valueAsNumber: true })} error={errors.loanAmount?.message} />
            <Input label="Loan Tenure (months)" type="number" {...register('loanTenure', { valueAsNumber: true })} />

            <Input
              label="Interest Rate (%)"
              type="number"
              step="0.1"
              disabled
              {...register('interestRate', { valueAsNumber: true })}
              hint="Auto-assigned from loan type and income slab"
            />

            <Input
              label="Existing Monthly EMI"
              type="number"
              {...register('existingEmis', { valueAsNumber: true })}
            />

            <Input
              label="EMI:Income Ratio (%)"
              type="number"
              disabled
              {...register('emiIncomeRatio', { valueAsNumber: true })}
            />
          </div>
        </section>
      )}

      {renderStepVisibility(2) && (
        <section className="rounded-[32px] border border-white/70 bg-white/85 p-6 shadow-xl backdrop-blur-xl">
          <div className="mb-5 flex items-center justify-between">
            <div>
              <p className="text-xs uppercase tracking-[0.24em] text-neutral-500">Assets & Credit</p>
              <h3 className="mt-2 text-2xl font-semibold text-neutral-900">Financial Profile</h3>
            </div>
            <div className="rounded-2xl bg-secondary-50 p-3 text-secondary-600">
              <Landmark className="h-6 w-6" />
            </div>
          </div>

          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
            <Input
              label="Residential Assets Value"
              type="number"
              {...register('residentialAssetsValue', { valueAsNumber: true })}
            />

            <Input
              label="Commercial Assets Value"
              type="number"
              {...register('commercialAssetsValue', { valueAsNumber: true })}
            />

            <Input
              label="Bank Balance"
              type="number"
              {...register('bankBalance', { valueAsNumber: true })}
            />

            <Input
              label="Total Assets (Auto)"
              type="number"
              disabled
              {...register('totalAssets', { valueAsNumber: true })}
            />

            <Input
              label="Total Liabilities"
              type="number"
              {...register('liabilities', { valueAsNumber: true })}
            />

            <Input label="CIBIL Score *" type="number" required {...register('cibilScore', { valueAsNumber: true })} error={errors.cibilScore?.message} />

            <div className="rounded-lg bg-blue-50 p-3 border border-blue-200">
              <p className="text-sm font-medium text-blue-900">
                Category: <span className="font-bold">{getCreditCategory(values.cibilScore)}</span>
              </p>
            </div>

            <Input
              label="Total Loans"
              type="number"
              {...register('totalLoans', { valueAsNumber: true })}
            />

            <Input
              label="Active Loans"
              type="number"
              {...register('activeLoans', { valueAsNumber: true })}
            />

            <Input
              label="Closed Loans"
              type="number"
              {...register('closedLoans', { valueAsNumber: true })}
            />

            <Input
              label="Missed Payments"
              type="number"
              {...register('missedPayments', { valueAsNumber: true })}
            />

            <Input
              label="Loan:Income Ratio (%)"
              type="number"
              disabled
              {...register('loanIncomeRatio', { valueAsNumber: true })}
            />

            <Input
              label="Debt:Income Ratio (%)"
              type="number"
              disabled
              {...register('debtToIncomeRatio', { valueAsNumber: true })}
            />
          </div>
        </section>
      )}

      {renderStepVisibility(3) && (
        <section className="rounded-[32px] border border-white/70 bg-white/85 p-6 shadow-xl backdrop-blur-xl">
          <div className="mb-6 flex items-center justify-between">
            <div>
              <p className="text-xs uppercase tracking-[0.24em] text-neutral-500">Supporting Documents</p>
              <h3 className="mt-2 text-2xl font-semibold text-neutral-900">Upload Documents</h3>
            </div>
            <div className="rounded-2xl bg-purple-50 p-3 text-purple-600">
              <FileText className="h-6 w-6" />
            </div>
          </div>

          <div className="grid gap-4 lg:grid-cols-3">
            {['Income proof', 'Payslip', 'Property proof'].map((documentType) => (
              <div key={documentType} className="rounded-2xl border border-neutral-200 bg-neutral-50 p-4">
                <p className="text-sm font-semibold text-neutral-900">{documentType}</p>
                <p className="mt-1 text-xs text-neutral-500">PDF, JPG, or PNG up to 5MB</p>
              </div>
            ))}
          </div>

          <div className="mt-5">
            <FileUploadArea
              acceptedFormats={['pdf', 'jpg', 'png']}
              maxSize={5 * 1024 * 1024}
              onFileSelect={async (file) => {
                setSelectedFile(file)
              }}
            />
          </div>

          {selectedFile && (
            <div className="mt-4 rounded-lg border border-green-200 bg-green-50 px-4 py-3">
              <p className="text-sm text-green-900">
                <span className="font-medium">Selected:</span> {selectedFile.name}
              </p>
            </div>
          )}
        </section>
      )}

      {renderStepVisibility(4) && (
        <section className="rounded-[32px] border border-white/70 bg-white/85 p-6 shadow-xl backdrop-blur-xl">
          <div className="mb-6">
            <p className="text-xs uppercase tracking-[0.24em] text-neutral-500">Application Summary</p>
            <h3 className="mt-2 text-2xl font-semibold text-neutral-900">Review Your Details</h3>
          </div>

          <div className="space-y-6">
            <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
              <div className="rounded-lg bg-gray-50 p-4">
                <p className="text-sm font-medium text-gray-600">Applicant</p>
                <p className="mt-1 text-lg font-semibold text-gray-900">
                  {values.firstName} {values.lastName}
                </p>
                <p className="mt-1 text-sm text-gray-600">{values.email}</p>
              </div>

              <div className="rounded-lg bg-gray-50 p-4">
                <p className="text-sm font-medium text-gray-600">Loan Details</p>
                <p className="mt-1 text-lg font-semibold text-gray-900">
                  ₹{values.loanAmount?.toLocaleString() || 'N/A'}
                </p>
                <p className="mt-1 text-sm text-gray-600">
                  {values.loanPurpose || 'N/A'} • {values.loanTenure} months
                </p>
                <p className="mt-1 text-sm text-gray-600">
                  Interest: {values.interestRate || 0}% auto-assigned
                </p>
              </div>

              <div className="rounded-lg bg-gray-50 p-4">
                <p className="text-sm font-medium text-gray-600">Income</p>
                <p className="mt-1 text-lg font-semibold text-gray-900">
                  ₹{monthlyIncome?.toLocaleString() || 'N/A'}/month
                </p>
                <p className="mt-1 text-sm text-gray-600">
                  EMI Ratio: {(emiIncomeRatio ?? 0).toFixed(1)}%
                </p>
                <p className="mt-1 text-sm text-gray-600">
                  Annual: ₹{Math.round(annualIncome || 0).toLocaleString()}
                </p>
              </div>

              <div className="rounded-lg bg-gray-50 p-4">
                <p className="text-sm font-medium text-gray-600">Credit Score</p>
                <p className="mt-1 text-lg font-semibold text-gray-900">
                  {values.cibilScore || 'N/A'}
                </p>
                <p className="mt-1 text-sm text-gray-600">
                  {getCreditCategory(values.cibilScore)}
                </p>
                <p className="mt-1 text-sm text-gray-600">
                  Credit utilization: {(MEAN_CREDIT_UTILIZATION_RATIO * 100).toFixed(0)}% mean value
                </p>
              </div>
            </div>

            <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
              <PreviewMetric label="Region" value={values.region ? String(values.region).replace('_', '-') : 'N/A'} />
              <PreviewMetric label="Total Assets" value={`₹${Math.round(totalAssets).toLocaleString()}`} />
              <PreviewMetric label="Loan:Income" value={`${(loanIncomeRatio ?? 0).toFixed(1)}%`} />
            </div>

            <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
              <div className="rounded-2xl border border-neutral-200 bg-white p-4">
                <p className="text-xs uppercase tracking-[0.18em] text-neutral-500">Identity and Contact</p>
                <div className="mt-3 space-y-2 text-sm text-neutral-700">
                  <p><span className="font-medium text-neutral-900">Name:</span> {displayValue(`${values.firstName || ''} ${values.lastName || ''}`.trim())}</p>
                  <p><span className="font-medium text-neutral-900">Email:</span> {displayValue(values.email)}</p>
                  <p><span className="font-medium text-neutral-900">Phone:</span> {displayValue(values.phone)}</p>
                  <p><span className="font-medium text-neutral-900">Age:</span> {displayValue(values.age)}</p>
                  <p><span className="font-medium text-neutral-900">Gender:</span> {displayValue(values.gender)}</p>
                  <p><span className="font-medium text-neutral-900">Marital Status:</span> {displayValue(values.maritalStatus)}</p>
                  <p><span className="font-medium text-neutral-900">City:</span> {displayValue(values.city)}</p>
                  <p><span className="font-medium text-neutral-900">Region:</span> {displayValue(values.region)}</p>
                </div>
              </div>

              <div className="rounded-2xl border border-neutral-200 bg-white p-4">
                <p className="text-xs uppercase tracking-[0.18em] text-neutral-500">Employment and Income</p>
                <div className="mt-3 space-y-2 text-sm text-neutral-700">
                  <p><span className="font-medium text-neutral-900">Employment Type:</span> {displayValue(values.employmentType)}</p>
                  <p><span className="font-medium text-neutral-900">Years of Employment:</span> {displayValue(values.yearsOfEmployment)}</p>
                  <p><span className="font-medium text-neutral-900">Monthly Income:</span> ₹{Math.round(monthlyIncome || 0).toLocaleString()}</p>
                  <p><span className="font-medium text-neutral-900">Annual Income:</span> ₹{Math.round(annualIncome || 0).toLocaleString()}</p>
                  <p><span className="font-medium text-neutral-900">EMI:</span> ₹{Math.round(values.emi || 0).toLocaleString()}</p>
                  <p><span className="font-medium text-neutral-900">Existing EMI:</span> ₹{Math.round(values.existingEmis || 0).toLocaleString()}</p>
                  <p><span className="font-medium text-neutral-900">EMI:Income Ratio:</span> {(emiIncomeRatio ?? 0).toFixed(2)}%</p>
                  <p><span className="font-medium text-neutral-900">Debt:Income Ratio:</span> {(debtToIncomeRatio ?? 0).toFixed(2)}%</p>
                </div>
              </div>

              <div className="rounded-2xl border border-neutral-200 bg-white p-4">
                <p className="text-xs uppercase tracking-[0.18em] text-neutral-500">Loan Request</p>
                <div className="mt-3 space-y-2 text-sm text-neutral-700">
                  <p><span className="font-medium text-neutral-900">Purpose:</span> {displayValue(values.loanPurpose)}</p>
                  <p><span className="font-medium text-neutral-900">Amount:</span> ₹{Math.round(values.loanAmount || 0).toLocaleString()}</p>
                  <p><span className="font-medium text-neutral-900">Tenure:</span> {displayValue(values.loanTenure)} months</p>
                  <p><span className="font-medium text-neutral-900">Interest:</span> {displayValue(values.interestRate)}%</p>
                  <p><span className="font-medium text-neutral-900">Loan:Income Ratio:</span> {(loanIncomeRatio ?? 0).toFixed(2)}%</p>
                </div>
              </div>

              <div className="rounded-2xl border border-neutral-200 bg-white p-4">
                <p className="text-xs uppercase tracking-[0.18em] text-neutral-500">Assets and Credit</p>
                <div className="mt-3 space-y-2 text-sm text-neutral-700">
                  <p><span className="font-medium text-neutral-900">Residential Assets:</span> ₹{Math.round(values.residentialAssetsValue || 0).toLocaleString()}</p>
                  <p><span className="font-medium text-neutral-900">Commercial Assets:</span> ₹{Math.round(values.commercialAssetsValue || 0).toLocaleString()}</p>
                  <p><span className="font-medium text-neutral-900">Bank Balance:</span> ₹{Math.round(values.bankBalance || 0).toLocaleString()}</p>
                  <p><span className="font-medium text-neutral-900">Total Assets:</span> ₹{Math.round(totalAssets).toLocaleString()}</p>
                  <p><span className="font-medium text-neutral-900">Liabilities:</span> ₹{Math.round(values.liabilities || 0).toLocaleString()}</p>
                  <p><span className="font-medium text-neutral-900">CIBIL Score:</span> {displayValue(values.cibilScore)}</p>
                  <p><span className="font-medium text-neutral-900">Total Loans:</span> {displayValue(values.totalLoans)}</p>
                  <p><span className="font-medium text-neutral-900">Missed Payments:</span> {displayValue(values.missedPayments)}</p>
                </div>
              </div>
            </div>

            <div className="rounded-lg border-l-4 border-blue-500 bg-blue-50 p-4">
              <div className="flex gap-3">
                <AlertCircle className="h-5 w-5 text-blue-600 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="font-medium text-blue-900">Please review your application</p>
                  <p className="mt-1 text-sm text-blue-800">
                    Ensure all information is accurate before submission. You can navigate back to correct any details.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>
      )}

      {isMultiStep && (
        <div className="flex items-center justify-between gap-3">
          <Button
            type="button"
            variant="ghost"
            onClick={() => setCurrentStep((step) => Math.max(step - 1, 0))}
            disabled={currentStep === 0}
            className="rounded-xl"
          >
            ← Previous
          </Button>

          <div className="flex gap-2">
            {steps.map((_, idx) => (
              <div
                key={idx}
                className={`h-2 flex-1 rounded-full transition-colors ${
                  idx <= currentStep ? 'bg-primary-500' : 'bg-gray-200'
                }`}
              />
            ))}
          </div>

          {currentStep < steps.length - 1 && (
            <Button
              type="button"
              variant="secondary"
              onClick={() => setCurrentStep((step) => Math.min(step + 1, steps.length - 1))}
              className="rounded-xl"
            >
              Next →
            </Button>
          )}
        </div>
      )}

      <Button
        variant="primary"
        size="lg"
        type="submit"
        isLoading={isSubmitting || isLoading}
        fullWidth
        disabled={!isValid || (isMultiStep && currentStep !== steps.length - 1)}
        className="rounded-2xl py-4 shadow-lg"
      >
        {isSubmitting || isLoading ? (
          <>
            <div className="mr-2 inline-block h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
            Submitting...
          </>
        ) : (
          'Submit Application'
        )}
      </Button>
    </form>
  )
}

function PreviewMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl bg-white/85 px-4 py-3">
      <p className="text-xs uppercase tracking-[0.18em] text-neutral-500">{label}</p>
      <p className="mt-1 text-lg font-semibold text-neutral-900">{value}</p>
    </div>
  )
}
