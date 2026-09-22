# 🏗️ SMARTLEND TYPESCRIPT DEFINITIONS & COMPONENT SPECS

## 📦 Core Type Definitions

### `types/application.ts`

```ts
// ============================================
// LOAN APPLICATION DATA MODEL
// ============================================

export type ApplicationStatus = 
  | 'draft' 
  | 'submitted' 
  | 'processing' 
  | 'approved' 
  | 'rejected' 
  | 'deferred'

export type DecisionType = 'approved' | 'rejected' | 'deferred'

export type ConfidenceLevel = 'low' | 'medium' | 'high'

export type LoanPurpose = 
  | 'home' 
  | 'auto' 
  | 'personal' 
  | 'business' 
  | 'education'

export type EmploymentType = 
  | 'salaried' 
  | 'self-employed' 
  | 'business' 
  | 'retired'

// Core domain entity
export interface LoanApplication {
  // Metadata
  id: string                              // UUID v4
  createdAt: string                       // ISO 8601
  updatedAt: string                       // ISO 8601
  status: ApplicationStatus
  
  // Applicant info
  applicantId: string
  applicantName: string
  email: string
  phone: string
  
  // Loan details
  loanAmount: number                      // In INR, minimum 100k
  loanPurpose: LoanPurpose
  loanTenure: number                      // Months
  interestRate?: number                   // Optional, from backend
  
  // Financial snapshot
  applicationData: {
    // Income & employment
    monthlyIncome: number                 // Gross monthly
    emi: number                           // Existing EMI obligations
    employmentType: EmploymentType
    yearsOfEmployment?: number
    
    // Assets & liabilities
    assets: number                        // Total assets
    liabilities?: number
    
    // Credit profile
    creditScore?: number                  // CIBIL (300-900)
    creditHistory?: 'excellent' | 'good' | 'average' | 'poor'
    
    // Demographics
    age: number
    dependents?: number
    maritalStatus?: 'single' | 'married' | 'divorced' | 'widowed'
    residenceType?: 'owned' | 'rented' | 'with_family'
  }
  
  // Decision from ML system
  decision?: ApplicationDecision
  
  // Documents
  documents?: Document[]
}

export interface ApplicationDecision {
  // Core decision
  id: string
  status: DecisionType
  decidedAt: string                       // ISO 8601
  decidedBy: 'model' | 'human'
  
  // Model outputs
  riskScore: number                       // [0, 1] - probability of default
  cbessScore: number                      // [0, 100] - explainability index
  uncertainty: number                     // [0, 1] - model uncertainty
  confidence: ConfidenceLevel
  
  // Explanation
  explanation: string                     // Human-readable decision rationale
  positiveFactors: string[]               // Why loan might be approved
  negativeFactors: string[]               // Risk factors
  
  // Feature importance
  featureImportance: FeatureContribution[]
  
  // Audit trail
  analystId?: string                      // If manually overridden
  analystNotes?: string
  modelVersion?: string                   // For reproducibility
}

export interface FeatureContribution {
  name: string                            // e.g., "Income", "EMI"
  impact: number                          // [-1, 1] - negative to positive
  value: number                           // Actual value from data
  baseValue?: number                      // For comparison
}

export interface Document {
  id: string
  fileName: string
  documentType: 'pdf' | 'csv' | 'jpg' | 'png'
  fileSize: number                        // Bytes
  uploadedAt: string
  extractedData?: Record<string, any>     // Parsed data from file
}

// Form data model (separate from persisted model)
export interface LoanApplicationFormData {
  loanAmount: number
  loanPurpose: LoanPurpose
  loanTenure: number
  monthlyIncome: number
  emi: number
  assets: number
  liabilities?: number
  creditScore?: number
  creditHistory?: string
  age: number
  dependents?: number
  employmentType: EmploymentType
  yearsOfEmployment?: number
  residenceType?: string
}

// Validation result
export interface ValidationResult {
  isValid: boolean
  errors: Record<string, string>
}
```

### `types/api.ts`

```ts
// ============================================
// API RESPONSE & REQUEST SCHEMAS
// ============================================

// Standard response envelope
export interface ApiResponse<T> {
  success: boolean
  data: T
  error?: {
    code: string
    message: string
    details?: Record<string, any>
    timestamp: string
  }
  timestamp: string
}

// Paginated response
export interface PaginatedResponse<T> {
  items: T[]
  total: number
  page: number
  pageSize: number
  hasMore: boolean
}

// Authentication
export interface LoginRequest {
  email: string
  password: string
}

export interface LoginResponse {
  token: string                           // JWT
  refreshToken: string
  user: {
    id: string
    email: string
    role: 'customer' | 'analyst' | 'admin'
    name: string
    organization?: string
  }
}

export interface AuthUser {
  id: string
  email: string
  role: 'customer' | 'analyst' | 'admin'
  name: string
  organization?: string
}

// Application endpoints
export interface CreateApplicationRequest {
  applicationData: LoanApplicationFormData
}

export interface CreateApplicationResponse {
  id: string
  status: ApplicationStatus
  createdAt: string
}

export interface UpdateApplicationRequest {
  applicationData: Partial<LoanApplicationFormData>
}

export interface FileUploadRequest {
  file: File
}

export interface FileUploadResponse {
  fileName: string
  documentType: string
  uploadedAt: string
  extractedData?: Record<string, any>
  fileSize: number
}

// Decision endpoints
export interface GetDecisionResponse {
  decision: ApplicationDecision
}

export interface ManualDecisionRequest {
  status: DecisionType
  notes: string
}

export interface ManualDecisionResponse {
  application: LoanApplication
  decidedAt: string
  decidedBy: string
}

// Dashboard metrics
export interface DashboardMetrics {
  totalApplications: number
  approved: number
  rejected: number
  deferred: number
  averageProcessingTime: number          // Seconds
  approvalRate: number                   // Percentage
  avgLoanAmount: number
  automationRate: number                 // Percentage of auto-decided
}

export interface ChartDataPoint {
  label: string
  value: number
  percentage?: number
}

export interface TrendDataPoint {
  date: string
  count: number
  approved?: number
  rejected?: number
  deferred?: number
}

// Error response
export interface ApiErrorResponse {
  code: string                            // e.g., 'VALIDATION_ERROR'
  message: string
  details?: {
    field?: string
    constraint?: string
    value?: any
  }[]
  timestamp: string
}
```

### `types/ui.ts`

```ts
// ============================================
// UI COMPONENT PROP TYPES
// ============================================

// Button
export interface ButtonProps 
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger'
  size?: 'sm' | 'md' | 'lg'
  isLoading?: boolean
  leftIcon?: React.ReactNode
  rightIcon?: React.ReactNode
}

// Card
export interface CardProps {
  title?: string
  description?: string
  children: React.ReactNode
  withGlass?: boolean
  className?: string
  footer?: React.ReactNode
}

// Input
export interface InputProps 
  extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string
  error?: string
  hint?: string
  icon?: React.ReactNode
  required?: boolean
}

// Select (Dropdown)
export interface SelectOption {
  value: string | number
  label: string
}

export interface SelectProps 
  extends Omit<React.SelectHTMLAttributes<HTMLSelectElement>, 'onChange'> {
  label?: string
  error?: string
  options: SelectOption[]
  onChange: (value: string | number) => void
  placeholder?: string
}

// Badge
export interface BadgeProps {
  status: 'approved' | 'rejected' | 'deferred' | 'pending'
  className?: string
}

// Modal
export interface ModalProps {
  isOpen: boolean
  onClose: () => void
  title?: string
  children: React.ReactNode
  size?: 'sm' | 'md' | 'lg'
  closeButton?: boolean
}

// Form Components
export interface FormProps {
  onSubmit: (data: any) => void | Promise<void>
  children: React.ReactNode
  isLoading?: boolean
}

export interface FormField {
  name: string
  label: string
  type: 'text' | 'number' | 'email' | 'select' | 'textarea' | 'file'
  required?: boolean
  placeholder?: string
  value?: any
  options?: SelectOption[]  // For select fields
  validation?: {
    min?: number
    max?: number
    pattern?: RegExp
  }
}

// Table
export interface TableColumn<T> {
  key: keyof T | string
  label: string
  sortable?: boolean
  format?: 'currency' | 'date' | 'percentage' | 'default'
  render?: (value: any, row: T) => React.ReactNode
  width?: string
}

export interface TableProps<T> {
  data: T[]
  columns: TableColumn<T>[]
  isLoading?: boolean
  onRowClick?: (row: T) => void
  sortBy?: keyof T
  sortOrder?: 'asc' | 'desc'
  pagination?: {
    pageSize: number
    currentPage: number
    onPageChange: (page: number) => void
  }
}

// KPI Card
export interface KPICardProps {
  label: string
  value: number | string
  format?: 'number' | 'currency' | 'percentage'
  trend?: {
    value: number
    direction: 'up' | 'down'
  }
  icon?: React.ComponentType<{ className?: string }>
  className?: string
}

// Chart
export interface ChartProps {
  type: 'pie' | 'line' | 'bar' | 'doughnut'
  data: any
  options?: any
  height?: number
  width?: number
}

// Feature Contribution Chart
export interface FeatureContributionChartProps {
  features: FeatureContribution[]
  maxFeatures?: number              // Show top N features
  isLoading?: boolean
}

// Decision Banner
export interface DecisionBannerProps {
  status: DecisionType
  riskScore: number
  confidence: ConfidenceLevel
  timestamp: string
  decidedBy: 'model' | 'human'
}

// Application Form
export interface ApplicationFormProps {
  initialData?: LoanApplicationFormData
  onSubmit: (data: LoanApplicationFormData) => Promise<void>
  isLoading?: boolean
  isMultiStep?: boolean
}

// File Upload
export interface FileUploadProps {
  onFileSelect: (file: File) => Promise<void>
  acceptedFormats?: string[]
  maxSize?: number                        // Bytes
  maxSizeLabel?: string
}

// Application Table Row
export interface ApplicationTableRowProps {
  application: LoanApplication
  onRowClick?: () => void
  isExpanded?: boolean
  onExpandToggle?: () => void
}

// Dialog / Confirmation
export interface ConfirmDialogProps {
  isOpen: boolean
  title: string
  message: string
  confirmText?: string
  cancelText?: string
  onConfirm: () => void | Promise<void>
  onCancel: () => void
  isDangerous?: boolean
}
```

---

## 🧩 COMPONENT SPECIFICATIONS

### 1. Button Component

```tsx
// components/common/Button.tsx

import React from 'react'
import { ButtonProps } from '@/types/ui'

const variantClasses = {
  primary: 'bg-primary-500 text-white hover:bg-primary-600 disabled:bg-neutral-300',
  secondary: 'bg-neutral-100 text-neutral-900 hover:bg-neutral-200 border border-neutral-300',
  ghost: 'text-neutral-600 hover:bg-neutral-100 hover:text-neutral-900',
  danger: 'bg-error text-white hover:bg-red-600 disabled:bg-neutral-300'
}

const sizeClasses = {
  sm: 'px-3 py-2 text-sm',
  md: 'px-4 py-2 text-base',
  lg: 'px-6 py-3 text-lg'
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ 
    variant = 'primary', 
    size = 'md', 
    isLoading = false,
    disabled = false,
    children,
    leftIcon,
    rightIcon,
    className,
    ...rest 
  }, ref) => {
    return (
      <button
        ref={ref}
        disabled={disabled || isLoading}
        className={`
          inline-flex items-center justify-center gap-2
          rounded-lg font-medium
          transition-colors duration-200
          focus:outline-2 focus:outline-offset-2 focus:outline-primary-500
          disabled:cursor-not-allowed
          ${variantClasses[variant]}
          ${sizeClasses[size]}
          ${className}
        `}
        {...rest}
      >
        {isLoading && <Spinner size={size} />}
        {leftIcon && !isLoading && <span>{leftIcon}</span>}
        {children}
        {rightIcon && <span>{rightIcon}</span>}
      </button>
    )
  }
)
Button.displayName = 'Button'
```

### 2. Card Component

```tsx
// components/common/Card.tsx

interface CardProps {
  title?: string
  description?: string
  children: React.ReactNode
  withGlass?: boolean
  className?: string
  footer?: React.ReactNode
}

export const Card: React.FC<CardProps> = ({
  title,
  description,
  children,
  withGlass = true,
  className = '',
  footer
}) => {
  return (
    <div
      className={`
        rounded-xl overflow-hidden
        border border-neutral-200
        bg-white
        ${withGlass ? 'backdrop-blur-md bg-opacity-80' : ''}
        shadow-md hover:shadow-lg transition-shadow
        ${className}
      `}
    >
      {(title || description) && (
        <div className="px-6 py-4 border-b border-neutral-200">
          {title && <h3 className="text-lg font-semibold text-neutral-900">{title}</h3>}
          {description && <p className="mt-1 text-sm text-neutral-600">{description}</p>}
        </div>
      )}
      
      <div className="px-6 py-4">
        {children}
      </div>
      
      {footer && (
        <div className="px-6 py-4 border-t border-neutral-200 bg-neutral-50">
          {footer}
        </div>
      )}
    </div>
  )
}
```

### 3. Form Input Component

```tsx
// components/forms/Input.tsx

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string
  error?: string
  hint?: string
  icon?: React.ReactNode
}

export const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ label, error, hint, icon, className = '', ...rest }, ref) => {
    return (
      <div className="w-full">
        {label && (
          <label className="block text-sm font-medium text-neutral-700 mb-2">
            {label}
            {rest.required && <span className="text-error ml-1">*</span>}
          </label>
        )}
        
        <div className="relative">
          <input
            ref={ref}
            className={`
              w-full px-4 py-2
              border rounded-lg
              text-neutral-900 placeholder-neutral-400
              focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent
              disabled:bg-neutral-50 disabled:text-neutral-500 disabled:cursor-not-allowed
              transition-colors
              ${error ? 'border-error' : 'border-neutral-300'}
              ${icon ? 'pl-10' : ''}
              ${className}
            `}
            {...rest}
          />
          {icon && (
            <span className="absolute left-3 top-1/2 -translate-y-1/2 text-neutral-500">
              {icon}
            </span>
          )}
        </div>
        
        {error && <p className="mt-1 text-sm text-error">{error}</p>}
        {hint && <p className="mt-1 text-sm text-neutral-500">{hint}</p>}
      </div>
    )
  }
)
Input.displayName = 'Input'
```

### 4. Badge Component (Status)

```tsx
// components/common/Badge.tsx

interface BadgeProps {
  status: 'approved' | 'rejected' | 'deferred' | 'pending'
  className?: string
}

const statusConfig = {
  approved: {
    bgColor: 'bg-green-50',
    textColor: 'text-green-700',
    borderColor: 'border-green-200',
    icon: '✓'
  },
  rejected: {
    bgColor: 'bg-red-50',
    textColor: 'text-red-700',
    borderColor: 'border-red-200',
    icon: '✗'
  },
  deferred: {
    bgColor: 'bg-amber-50',
    textColor: 'text-amber-700',
    borderColor: 'border-amber-200',
    icon: '⏳'
  },
  pending: {
    bgColor: 'bg-blue-50',
    textColor: 'text-blue-700',
    borderColor: 'border-blue-200',
    icon: '⏵'
  }
}

export const Badge: React.FC<BadgeProps> = ({ status, className = '' }) => {
  const config = statusConfig[status]
  const label = status.charAt(0).toUpperCase() + status.slice(1)
  
  return (
    <span
      className={`
        inline-flex items-center gap-1
        px-3 py-1 rounded-full
        border text-sm font-medium
        ${config.bgColor} ${config.textColor} ${config.borderColor}
        ${className}
      `}
    >
      <span>{config.icon}</span>
      {label}
    </span>
  )
}
```

### 5. Loan Application Form

```tsx
// components/forms/LoanApplicationForm.tsx

import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import * as z from 'zod'
import { LoanApplicationFormData } from '@/types/application'

const validationSchema = z.object({
  loanAmount: z.number().min(100000).max(10000000),
  loanPurpose: z.enum(['home', 'auto', 'personal', 'business']),
  loanTenure: z.number().min(12).max(360),
  monthlyIncome: z.number().min(15000).max(500000),
  emi: z.number().min(0),
  assets: z.number().min(0),
  age: z.number().min(21).max(70),
  dependents: z.number().min(0).optional()
})

interface LoanApplicationFormProps {
  onSubmit: (data: LoanApplicationFormData) => Promise<void>
  isLoading?: boolean
}

export const LoanApplicationForm: React.FC<LoanApplicationFormProps> = ({
  onSubmit,
  isLoading = false
}) => {
  const {
    register,
    handleSubmit,
    watch,
    formState: { errors, isSubmitting }
  } = useForm<LoanApplicationFormData>({
    resolver: zodResolver(validationSchema),
    mode: 'onChange'
  })

  const monthlyIncome = watch('monthlyIncome')
  const maxEMI = monthlyIncome ? Math.round(monthlyIncome * 0.4) : 0

  return (
    <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        
        {/* Loan Amount */}
        <Input
          label="Requested Loan Amount (₹)"
          type="number"
          placeholder="e.g., 500000"
          {...register('loanAmount', { valueAsNumber: true })}
          error={errors.loanAmount?.message}
        />

        {/* Loan Purpose */}
        <Select
          label="Loan Purpose"
          {...register('loanPurpose')}
          options={[
            { value: 'home', label: 'Home Loan' },
            { value: 'auto', label: 'Auto Loan' },
            { value: 'personal', label: 'Personal Loan' },
            { value: 'business', label: 'Business Loan' }
          ]}
          error={errors.loanPurpose?.message}
        />

        {/* Monthly Income */}
        <Input
          label="Monthly Income (₹)"
          type="number"
          placeholder="e.g., 50000"
          {...register('monthlyIncome', { valueAsNumber: true })}
          error={errors.monthlyIncome?.message}
        />

        {/* EMI */}
        <Input
          label={`Current EMI Obligations (₹) - Max: ₹${maxEMI.toLocaleString('en-IN')}`}
          type="number"
          placeholder="e.g., 10000"
          {...register('emi', { valueAsNumber: true })}
          error={errors.emi?.message}
          hint="Should not exceed 40% of monthly income"
        />

        {/* Assets */}
        <Input
          label="Total Assets (₹)"
          type="number"
          placeholder="e.g., 500000"
          {...register('assets', { valueAsNumber: true })}
          error={errors.assets?.message}
        />

        {/* Age */}
        <Input
          label="Age"
          type="number"
          placeholder="e.g., 35"
          {...register('age', { valueAsNumber: true })}
          error={errors.age?.message}
        />

        {/* Loan Tenure */}
        <Input
          label="Loan Tenure (Months)"
          type="number"
          placeholder="e.g., 60"
          {...register('loanTenure', { valueAsNumber: true })}
          error={errors.loanTenure?.message}
        />

        {/* Dependents */}
        <Input
          label="Number of Dependents"
          type="number"
          placeholder="e.g., 2"
          {...register('dependents', { valueAsNumber: true })}
        />
      </div>

      {/* File Upload */}
      <div>
        <label className="block text-sm font-medium text-neutral-700 mb-2">
          Upload Supporting Documents (Optional)
        </label>
        <FileUploadArea
          acceptedFormats={['pdf', 'csv']}
          maxSize={5 * 1024 * 1024}
        />
      </div>

      {/* Submit */}
      <Button
        variant="primary"
        size="lg"
        type="submit"
        isLoading={isSubmitting || isLoading}
        fullWidth
      >
        {isSubmitting || isLoading ? 'Submitting...' : 'Submit Application'}
      </Button>
    </form>
  )
}
```

### 6. Decision Explanation Component

```tsx
// components/sections/DecisionExplanation.tsx

import { ApplicationDecision } from '@/types/application'

interface DecisionExplanationProps {
  decision: ApplicationDecision
}

export const DecisionExplanation: React.FC<DecisionExplanationProps> = ({
  decision
}) => {
  return (
    <Card title="Decision Analysis" className="mt-6">
      <div className="space-y-6">
        
        {/* Main Explanation */}
        <div>
          <h4 className="font-semibold text-neutral-900 mb-2">Why this decision?</h4>
          <p className="text-neutral-700 leading-relaxed">
            {decision.explanation}
          </p>
        </div>

        {/* Positive Factors */}
        {decision.positiveFactors.length > 0 && (
          <div>
            <h4 className="font-semibold text-green-700 mb-2">✓ Positive Factors</h4>
            <ul className="space-y-1">
              {decision.positiveFactors.map((factor, idx) => (
                <li key={idx} className="text-sm text-neutral-700 flex items-start gap-2">
                  <span className="text-green-600 mt-1">✓</span>
                  <span>{factor}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Negative Factors */}
        {decision.negativeFactors.length > 0 && (
          <div>
            <h4 className="font-semibold text-red-700 mb-2">⚠ Risk Factors</h4>
            <ul className="space-y-1">
              {decision.negativeFactors.map((factor, idx) => (
                <li key={idx} className="text-sm text-neutral-700 flex items-start gap-2">
                  <span className="text-red-600 mt-1">⚠</span>
                  <span>{factor}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Decision Metadata */}
        <div className="pt-4 border-t border-neutral-200">
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <p className="text-neutral-600">Decided By</p>
              <p className="font-semibold text-neutral-900 capitalize">
                {decision.decidedBy === 'model' ? '🤖 AI Model' : '👤 Analyst'}
              </p>
            </div>
            <div>
              <p className="text-neutral-600">Decision Time</p>
              <p className="font-semibold text-neutral-900">
                {new Date(decision.decidedAt).toLocaleDateString()}
              </p>
            </div>
          </div>
        </div>
      </div>
    </Card>
  )
}
```

---

## 🔌 HOOKS SPECIFICATIONS

### `hooks/useFormValidation.ts`

```ts
export function useFormValidation(schema: ZodSchema) {
  const {
    register,
    handleSubmit,
    watch,
    formState: { errors, isSubmitting, isValid },
    reset
  } = useForm({
    resolver: zodResolver(schema),
    mode: 'onChange'
  })

  return { register, handleSubmit, watch, errors, isSubmitting, isValid, reset }
}
```

### `hooks/useApplicationData.ts`

```ts
export function useApplicationData(applicationId: string) {
  const [application, setApplication] = useState<LoanApplication | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchApplication = async () => {
      try {
        const response = await apiClient.get(`/applications/${applicationId}`)
        setApplication(response.data)
      } catch (err) {
        setError(err.message)
      } finally {
        setIsLoading(false)
      }
    }

    fetchApplication()
  }, [applicationId])

  return { application, isLoading, error }
}
```

---

**End of TypeScript Definitions**
