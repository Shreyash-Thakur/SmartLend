export type ApplicationStatus =
  | 'draft'
  | 'submitted'
  | 'processing'
  | 'approved'
  | 'rejected'
  | 'deferred'

export type DecisionType = 'approved' | 'rejected' | 'deferred'

export type ConfidenceLevel = 'low' | 'medium' | 'high'

export type LoanPurpose = 'home' | 'auto' | 'personal' | 'business' | 'education'

export type EmploymentType = 'salaried' | 'self-employed' | 'business' | 'retired'

export type Gender = 'male' | 'female' | 'other'

export type MaritalStatus = 'single' | 'married' | 'divorced' | 'widowed'

export type EducationLevel = 'high_school' | 'diploma' | 'graduate' | 'postgraduate' | 'doctorate'

export type Region =
  | 'rural'
  | 'urban'
  | 'semi_urban'

export interface FeatureContribution {
  name: string
  impact: number
  value: number
  baseValue?: number
}

export interface ApplicationDecision {
  id: string
  status: DecisionType
  decidedAt: string
  decidedBy: 'model' | 'human'
  riskScore: number
  cbessScore: number
  uncertainty: number
  confidence: ConfidenceLevel
  explanation: string
  positiveFactors: string[]
  negativeFactors: string[]
  featureImportance: FeatureContribution[]
  analystId?: string
  analystNotes?: string
  modelVersion?: string
}

export interface Document {
  id: string
  fileName: string
  documentType: 'pdf' | 'csv' | 'jpg' | 'png'
  fileSize: number
  uploadedAt: string
  extractedData?: Record<string, unknown>
}

export interface LoanApplication {
  id: string
  createdAt: string
  updatedAt: string
  status: ApplicationStatus
  source: 'seed' | 'customer'
  applicantId: string
  applicantName: string
  email: string
  phone: string
  loanAmount: number
  loanPurpose: LoanPurpose
  loanTenure: number
  interestRate?: number
  ml_prob?: number
  cbes_prob?: number
  cbes_score?: number
  confidence?: number
  finalDecision?: 'APPROVE' | 'REJECT' | 'DEFER'
  modelRecommendation?: DecisionType | 'submitted'
  manualDecisionApplied?: boolean
  applicationData: {
    firstName: string
    lastName: string
    gender: Gender
    maritalStatus?: MaritalStatus
    education?: EducationLevel
    monthlyIncome: number
    annualIncome?: number
    emi: number
    existingEmis?: number
    employmentType: EmploymentType
    yearsOfEmployment?: number
    assets: number
    residentialAssetsValue?: number
    commercialAssetsValue?: number
    bankBalance?: number
    totalAssets?: number
    liabilities?: number
    creditScore?: number
    cibilScore?: number
    creditHistory?: 'excellent' | 'good' | 'average' | 'poor'
    totalLoans?: number
    activeLoans?: number
    closedLoans?: number
    missedPayments?: number
    creditUtilizationRatio?: number
    emiIncomeRatio?: number
    loanIncomeRatio?: number
    debtToIncomeRatio?: number
    age: number
    dependents?: number
    residenceType?: 'owned' | 'rented' | 'with_family'
    region?: Region
    city?: string
  }
  decision?: ApplicationDecision
  documents?: Document[]
}

export interface LoanApplicationFormData {
  applicantId?: string
  firstName: string
  lastName: string
  email: string
  phone: string
  gender: Gender
  maritalStatus?: MaritalStatus
  education?: EducationLevel
  loanAmount: number
  loanPurpose: LoanPurpose
  loanTenure: number
  interestRate?: number
  monthlyIncome: number
  annualIncome?: number
  emi: number
  existingEmis?: number
  assets: number
  residentialAssetsValue?: number
  commercialAssetsValue?: number
  bankBalance?: number
  totalAssets?: number
  liabilities?: number
  creditScore?: number
  cibilScore?: number
  creditHistory?: 'excellent' | 'good' | 'average' | 'poor'
  totalLoans?: number
  activeLoans?: number
  closedLoans?: number
  missedPayments?: number
  creditUtilizationRatio?: number
  emiIncomeRatio?: number
  loanIncomeRatio?: number
  debtToIncomeRatio?: number
  age: number
  dependents?: number
  employmentType: EmploymentType
  yearsOfEmployment?: number
  residenceType?: 'owned' | 'rented' | 'with_family'
  region?: Region
  city?: string
}

export interface ValidationResult {
  isValid: boolean
  errors: Record<string, string>
}
