import type React from 'react'
import type {
  ConfidenceLevel,
  DecisionType,
  FeatureContribution,
  LoanApplication,
  LoanApplicationFormData,
} from '@/types/application'

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger'
  size?: 'sm' | 'md' | 'lg'
  isLoading?: boolean
  leftIcon?: React.ReactNode
  rightIcon?: React.ReactNode
  fullWidth?: boolean
}

export interface CardProps {
  title?: string
  description?: string
  children: React.ReactNode
  withGlass?: boolean
  className?: string
  footer?: React.ReactNode
}

export interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string
  error?: string
  hint?: string
  icon?: React.ReactNode
  required?: boolean
}

export interface TextareaProps extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {
  label?: string
  error?: string
  hint?: string
  required?: boolean
}

export interface SelectOption {
  value: string | number
  label: string
}

export interface SelectProps
  extends Omit<React.SelectHTMLAttributes<HTMLSelectElement>, 'onChange'> {
  label?: string
  error?: string
  options: SelectOption[]
  onChange?: (value: string | number) => void
  placeholder?: string
}

export interface BadgeProps {
  status:
    | 'draft'
    | 'approved'
    | 'rejected'
    | 'deferred'
    | 'pending'
    | 'submitted'
    | 'processing'
  className?: string
}

export interface ModalProps {
  isOpen: boolean
  onClose: () => void
  title?: string
  children: React.ReactNode
  size?: 'sm' | 'md' | 'lg'
  closeButton?: boolean
}

export interface KPICardProps {
  label: string
  value: number | string
  format?: 'number' | 'currency' | 'percentage'
  trend?: {
    value: number
    direction: 'up' | 'down'
  }
  className?: string
}

export interface FeatureContributionChartProps {
  features: FeatureContribution[]
  maxFeatures?: number
  isLoading?: boolean
}

export interface DecisionBannerProps {
  status: DecisionType
  riskScore: number
  confidence: ConfidenceLevel
  timestamp: string
  decidedBy: 'model' | 'human'
}

export interface ApplicationFormProps {
  initialData?: Partial<LoanApplicationFormData>
  onSubmit: (data: LoanApplicationFormData, file?: File) => Promise<void>
  isLoading?: boolean
  isMultiStep?: boolean
}

export interface FileUploadProps {
  onFileSelect: (file: File) => Promise<void> | void
  acceptedFormats?: string[]
  maxSize?: number
  maxSizeLabel?: string
}

export interface TableColumn<T> {
  key: keyof T | string
  label: string
  sortable?: boolean
  format?: 'currency' | 'date' | 'percentage' | 'default'
  render?: (value: unknown, row: T) => React.ReactNode
}

export interface ApplicationTableProps {
  data: LoanApplication[]
  isLoading?: boolean
  pageSize?: number
  onRowClick?: (application: LoanApplication) => void
  showApplicant?: boolean
  selectedIds?: Set<string>
  onToggleSelect?: (id: string) => void
  hideMetrics?: boolean
}
