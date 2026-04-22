import React, { useEffect, useMemo, useState } from 'react'
import { Controller } from 'react-hook-form'
import { Landmark, MapPin, Wallet } from 'lucide-react'
import { Button, Input, Select } from '@/components/common'
import type { LoanApplicationFormData } from '@/types/application'
import type { ApplicationFormProps } from '@/types/ui'
import { loanApplicationSchema, useFormValidation } from '@/hooks/useFormValidation'
import { FileUploadArea } from './FileUploadArea'

const defaultValues: LoanApplicationFormData = {
  applicantId: '',
  firstName: '',
  lastName: '',
  email: '',
  phone: '',
  gender: 'male',
  maritalStatus: 'single',
  education: 'graduate',
  age: 30,
  dependents: 0,
  employmentType: 'salaried',
  yearsOfEmployment: 5,
  monthlyIncome: 50000,
  annualIncome: 600000,
  loanPurpose: 'business',
  loanAmount: 500000,
  loanTenure: 60,
  interestRate: 11.5,
  emi: 8000,
  existingEmis: 0,
  residentialAssetsValue: 350000,
  commercialAssetsValue: 0,
  bankBalance: 180000,
  totalAssets: 530000,
  assets: 530000,
  liabilities: 100000,
  creditScore: 720,
  creditHistory: 'good',
  totalLoans: 2,
  activeLoans: 1,
  closedLoans: 1,
  missedPayments: 0,
  creditUtilizationRatio: 28,
  emiIncomeRatio: 16,
  loanIncomeRatio: 83,
  debtToIncomeRatio: 22,
  residenceType: 'owned',
  region: 'west',
  city: '',
}

const steps = [
  'Identity',
  'Income',
  'Assets',
  'Credit',
  'Documents',
]

export const LoanApplicationForm: React.FC<ApplicationFormProps> = ({
  onSubmit,
  initialData,
  isLoading = false,
  isMultiStep = true,
}) => {
  const storageKey = 'smartlend.loan-application.draft.v2'
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
  const monthlyIncome = values.monthlyIncome || 0
  const annualIncome = values.annualIncome || monthlyIncome * 12
  const totalAssets =
    (values.residentialAssetsValue || 0) +
    (values.commercialAssetsValue || 0) +
    (values.bankBalance || 0)
  const totalEmis = (values.emi || 0) + (values.existingEmis || 0)
  const emiIncomeRatio = monthlyIncome ? (totalEmis / monthlyIncome) * 100 : 0
  const loanIncomeRatio = annualIncome ? (values.loanAmount / annualIncome) * 100 : 0
  const debtToIncomeRatio =
    annualIncome ? (((values.liabilities || 0) + totalEmis * 12) / annualIncome) * 100 : 0
  const maxEmi = monthlyIncome ? Math.round(monthlyIncome * 0.4) : 0

  useEffect(() => {
    reset({
      ...defaultValues,
      ...savedDraft,
      ...initialData,
    })
  }, [initialData, reset, savedDraft])

  useEffect(() => {
    setValue('annualIncome', Math.round(annualIncome))
    setValue('totalAssets', Math.round(totalAssets))
    setValue('assets', Math.round(totalAssets))
    setValue('emiIncomeRatio', Number(emiIncomeRatio.toFixed(2)))
    setValue('loanIncomeRatio', Number(loanIncomeRatio.toFixed(2)))
    setValue('debtToIncomeRatio', Number(debtToIncomeRatio.toFixed(2)))
  }, [annualIncome, debtToIncomeRatio, emiIncomeRatio, loanIncomeRatio, setValue, totalAssets])

  useEffect(() => {
    const subscription = watch((value) => {
      localStorage.setItem(storageKey, JSON.stringify(value))
    })
    return () => subscription.unsubscribe()
  }, [watch])

  const submitForm = async (data: LoanApplicationFormData) => {
    await onSubmit(data)
    localStorage.removeItem(storageKey)
    setSelectedFile(null)
    if (isMultiStep) {
      setCurrentStep(0)
    }
  }

  const renderStepVisibility = (index: number) => !isMultiStep || currentStep === index

  return (
    <form onSubmit={handleSubmit(submitForm)} className="space-y-8">
      {isMultiStep && (
        <div className="rounded-[28px] border border-white/70 bg-white/80 p-5 shadow-lg backdrop-blur-xl">
          <div className="grid gap-3 md:grid-cols-5">
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
              <p className="text-xs uppercase tracking-[0.24em] text-neutral-500">Applicant Profile</p>
              <h3 className="mt-2 text-2xl font-semibold text-neutral-900">Identity and residency</h3>
            </div>
            <div className="rounded-2xl bg-primary-50 p-3 text-primary-600">
              <MapPin className="h-6 w-6" />
            </div>
          </div>

          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
            <Input label="Applicant ID" placeholder="Auto-generated if blank" {...register('applicantId')} />
            <Input label="First Name" required {...register('firstName')} error={errors.firstName?.message} />
            <Input label="Last Name" required {...register('lastName')} error={errors.lastName?.message} />
            <Input label="Email" type="email" required {...register('email')} error={errors.email?.message} />
            <Input label="Phone" required {...register('phone')} error={errors.phone?.message} />
            <Input label="City" required {...register('city')} error={errors.city?.message} />

            <Controller
              control={control}
              name="gender"
              render={({ field }) => (
                <Select
                  label="Gender"
                  options={[
                    { value: 'male', label: 'Male' },
                    { value: 'female', label: 'Female' },
                    { value: 'other', label: 'Other' },
                  ]}
                  value={field.value}
                  onChange={field.onChange}
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
                    { value: 'single', label: 'Single' },
                    { value: 'married', label: 'Married' },
                    { value: 'divorced', label: 'Divorced' },
                    { value: 'widowed', label: 'Widowed' },
                  ]}
                  value={field.value || ''}
                  onChange={field.onChange}
                />
              )}
            />

            <Controller
              control={control}
              name="education"
              render={({ field }) => (
                <Select
                  label="Education"
                  options={[
                    { value: 'high_school', label: 'High School' },
                    { value: 'diploma', label: 'Diploma' },
                    { value: 'graduate', label: 'Graduate' },
                    { value: 'postgraduate', label: 'Postgraduate' },
                    { value: 'doctorate', label: 'Doctorate' },
                  ]}
                  value={field.value || ''}
                  onChange={field.onChange}
                />
              )}
            />

            <Input label="Age" type="number" required {...register('age', { valueAsNumber: true })} error={errors.age?.message} />
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
                  label="Region"
                  options={[
                    { value: 'north', label: 'North' },
                    { value: 'south', label: 'South' },
                    { value: 'east', label: 'East' },
                    { value: 'west', label: 'West' },
                    { value: 'central', label: 'Central' },
                    { value: 'north_east', label: 'North East' },
                  ]}
                  value={field.value || ''}
                  onChange={field.onChange}
                />
              )}
            />

            <Controller
              control={control}
              name="residenceType"
              render={({ field }) => (
                <Select
                  label="Residence Type"
                  options={[
                    { value: 'owned', label: 'Owned' },
                    { value: 'rented', label: 'Rented' },
                    { value: 'with_family', label: 'With Family' },
                  ]}
                  value={field.value || ''}
                  onChange={field.onChange}
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
              <p className="text-xs uppercase tracking-[0.24em] text-neutral-500">Income Layer</p>
              <h3 className="mt-2 text-2xl font-semibold text-neutral-900">Employment and loan request</h3>
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
                  label="Employment Type"
                  options={[
                    { value: 'salaried', label: 'Salaried' },
                    { value: 'self-employed', label: 'Self Employed' },
                    { value: 'business', label: 'Business Owner' },
                    { value: 'retired', label: 'Retired' },
                  ]}
                  value={field.value}
                  onChange={field.onChange}
                />
              )}
            />
            <Input
              label="Years Employed"
              type="number"
              {...register('yearsOfEmployment', { valueAsNumber: true })}
              error={errors.yearsOfEmployment?.message}
            />
            <Input
              label="Monthly Income"
              type="number"
              required
              {...register('monthlyIncome', { valueAsNumber: true })}
              error={errors.monthlyIncome?.message}
            />
            <Input label="Annual Income" type="number" value={annualIncome} readOnly icon={<Wallet className="h-4 w-4" />} />
            <Input
              label="Loan Amount"
              type="number"
              required
              {...register('loanAmount', { valueAsNumber: true })}
              error={errors.loanAmount?.message}
            />
            <Input
              label="Loan Term (Months)"
              type="number"
              required
              {...register('loanTenure', { valueAsNumber: true })}
              error={errors.loanTenure?.message}
            />
            <Controller
              control={control}
              name="loanPurpose"
              render={({ field }) => (
                <Select
                  label="Loan Type"
                  options={[
                    { value: 'home', label: 'Home Loan' },
                    { value: 'auto', label: 'Auto Loan' },
                    { value: 'personal', label: 'Personal Loan' },
                    { value: 'business', label: 'Business Loan' },
                    { value: 'education', label: 'Education Loan' },
                  ]}
                  value={field.value}
                  onChange={field.onChange}
                />
              )}
            />
            <Input
              label="Interest Rate (%)"
              type="number"
              step="0.1"
              {...register('interestRate', { valueAsNumber: true })}
              error={errors.interestRate?.message}
            />
            <Input
              label={`Primary EMI (Max ${maxEmi.toLocaleString('en-IN')})`}
              type="number"
              required
              {...register('emi', { valueAsNumber: true })}
              error={errors.emi?.message}
            />
            <Input
              label="Existing EMIs"
              type="number"
              {...register('existingEmis', { valueAsNumber: true })}
              error={errors.existingEmis?.message}
            />
          </div>
        </section>
      )}

      {renderStepVisibility(2) && (
        <section className="rounded-[32px] border border-white/70 bg-white/85 p-6 shadow-xl backdrop-blur-xl">
          <div className="mb-5 flex items-center justify-between">
            <div>
              <p className="text-xs uppercase tracking-[0.24em] text-neutral-500">Balance Sheet</p>
              <h3 className="mt-2 text-2xl font-semibold text-neutral-900">Assets and liabilities</h3>
            </div>
            <div className="rounded-2xl bg-yellow-100 p-3 text-yellow-600">
              <Landmark className="h-6 w-6" />
            </div>
          </div>

          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
            <Input
              label="Residential Assets Value"
              type="number"
              {...register('residentialAssetsValue', { valueAsNumber: true })}
              error={errors.residentialAssetsValue?.message}
            />
            <Input
              label="Commercial Assets Value"
              type="number"
              {...register('commercialAssetsValue', { valueAsNumber: true })}
              error={errors.commercialAssetsValue?.message}
            />
            <Input
              label="Bank Balance"
              type="number"
              {...register('bankBalance', { valueAsNumber: true })}
              error={errors.bankBalance?.message}
            />
            <Input label="Total Assets" type="number" value={Math.round(totalAssets)} readOnly />
            <Input label="Assets Mirror Field" type="number" value={Math.round(totalAssets)} readOnly />
            <Input
              label="Liabilities"
              type="number"
              {...register('liabilities', { valueAsNumber: true })}
              error={errors.liabilities?.message}
            />
          </div>

          <div className="mt-6 rounded-[28px] bg-neutral-900 p-6 text-white shadow-lg">
            <div className="grid gap-4 md:grid-cols-3">
              <PreviewMetric label="EMI / Income Ratio" value={`${emiIncomeRatio.toFixed(1)}%`} />
              <PreviewMetric label="Loan / Income Ratio" value={`${loanIncomeRatio.toFixed(1)}%`} />
              <PreviewMetric label="Debt / Income Ratio" value={`${debtToIncomeRatio.toFixed(1)}%`} />
            </div>
          </div>
        </section>
      )}

      {renderStepVisibility(3) && (
        <section className="rounded-[32px] border border-white/70 bg-white/85 p-6 shadow-xl backdrop-blur-xl">
          <div className="mb-5">
            <p className="text-xs uppercase tracking-[0.24em] text-neutral-500">Credit Layer</p>
            <h3 className="mt-2 text-2xl font-semibold text-neutral-900">Bureau and repayment history</h3>
          </div>

          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
            <Input
              label="CIBIL Score"
              type="number"
              {...register('creditScore', { valueAsNumber: true })}
              error={errors.creditScore?.message}
            />
            <Controller
              control={control}
              name="creditHistory"
              render={({ field }) => (
                <Select
                  label="Credit History"
                  options={[
                    { value: 'excellent', label: 'Excellent' },
                    { value: 'good', label: 'Good' },
                    { value: 'average', label: 'Average' },
                    { value: 'poor', label: 'Poor' },
                  ]}
                  value={field.value || ''}
                  onChange={field.onChange}
                />
              )}
            />
            <Input label="Total Loans" type="number" {...register('totalLoans', { valueAsNumber: true })} />
            <Input label="Active Loans" type="number" {...register('activeLoans', { valueAsNumber: true })} />
            <Input label="Closed Loans" type="number" {...register('closedLoans', { valueAsNumber: true })} />
            <Input label="Missed Payments" type="number" {...register('missedPayments', { valueAsNumber: true })} />
            <Input
              label="Credit Utilization Ratio (%)"
              type="number"
              step="0.1"
              {...register('creditUtilizationRatio', { valueAsNumber: true })}
            />
            <Input label="EMI / Income Ratio (%)" type="number" value={emiIncomeRatio.toFixed(2)} readOnly />
            <Input label="Loan / Income Ratio (%)" type="number" value={loanIncomeRatio.toFixed(2)} readOnly />
            <Input label="Debt / Income Ratio (%)" type="number" value={debtToIncomeRatio.toFixed(2)} readOnly />
          </div>
        </section>
      )}

      {renderStepVisibility(4) && (
        <section className="rounded-[32px] border border-white/70 bg-white/85 p-6 shadow-xl backdrop-blur-xl">
          <div className="mb-6 flex items-center justify-between">
            <div>
              <p className="text-xs uppercase tracking-[0.24em] text-neutral-500">Documents and review</p>
              <h3 className="mt-2 text-2xl font-semibold text-neutral-900">Final verification</h3>
            </div>
            <div className="rounded-2xl bg-primary-500 px-4 py-2 text-sm font-semibold text-white">
              Session-only draft
            </div>
          </div>

          <div className="grid gap-6 lg:grid-cols-[1.2fr_0.8fr]">
            <div className="space-y-4">
              <FileUploadArea
                acceptedFormats={['pdf', 'csv']}
                maxSize={5 * 1024 * 1024}
                onFileSelect={async (file) => {
                  setSelectedFile(file)
                }}
              />
              {selectedFile && (
                <div className="rounded-2xl border border-primary-100 bg-primary-50 px-4 py-3 text-sm text-primary-900">
                  Ready to attach after creation: {selectedFile.name}
                </div>
              )}
            </div>

            <div className="rounded-[28px] bg-gradient-to-br from-[#f2f6ee] via-[#edf7f2] to-[#f8fafc] p-5 shadow-inner">
              <p className="text-xs uppercase tracking-[0.24em] text-neutral-500">Underwriting preview</p>
              <div className="mt-5 space-y-4">
                <PreviewMetric label="Applicant" value={`${values.firstName || '-'} ${values.lastName || ''}`.trim()} />
                <PreviewMetric label="Loan Request" value={`₹${Number(values.loanAmount || 0).toLocaleString('en-IN')}`} />
                <PreviewMetric label="Monthly Income" value={`₹${Number(monthlyIncome).toLocaleString('en-IN')}`} />
                <PreviewMetric label="CIBIL Score" value={values.creditScore ? String(values.creditScore) : 'Pending'} />
                <PreviewMetric
                  label="Auto Confidence Preview"
                  value={emiIncomeRatio < 25 && (values.creditScore || 0) > 700 ? 'High' : emiIncomeRatio < 40 ? 'Medium' : 'Low'}
                />
              </div>
            </div>
          </div>
        </section>
      )}

      {isMultiStep && (
        <div className="flex items-center justify-between">
          <Button
            type="button"
            variant="ghost"
            onClick={() => setCurrentStep((step) => Math.max(step - 1, 0))}
            disabled={currentStep === 0}
          >
            Previous
          </Button>
          <Button
            type="button"
            variant="secondary"
            onClick={() => setCurrentStep((step) => Math.min(step + 1, steps.length - 1))}
            disabled={currentStep === steps.length - 1}
          >
            Next
          </Button>
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
        {isSubmitting || isLoading ? 'Submitting Application...' : 'Create Loan Application'}
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
