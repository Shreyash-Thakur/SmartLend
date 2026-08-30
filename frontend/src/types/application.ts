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
  allModelPredictions?: Record<string, number>
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
  analystNotes?: string
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

// ---------------------------------------------------------------------------
// Redesigned short application form (docs/FORM-REDESIGN.md)
// ---------------------------------------------------------------------------
// The applicant is an existing bank customer, so the form asks only for what
// the bank cannot answer itself. Everything else — age, gender, region,
// income, employment, and the whole bureau block including the credit score —
// is resolved server-side from `customer_id`.

export type LoanPurposeShort =
  | 'home_improvement'
  | 'education'
  | 'medical'
  | 'debt_consolidation'
  | 'business'
  | 'vehicle'
  | 'wedding'
  | 'other'

export type RepaymentSource = 'salary' | 'business_income' | 'rental' | 'other'

export type CollateralType = 'none' | 'property' | 'gold' | 'fixed_deposit' | 'vehicle'

export type GuarantorRelationship = 'spouse' | 'parent' | 'sibling' | 'employer' | 'other'

/** The read-only "we already have this" block returned by
 *  GET /api/customers/{customer_id}/profile. */
export interface CustomerProfile {
  customer_id: string
  found: boolean
  age: number | null
  gender: string | null
  dependents: number | null
  marital_status: string | null
  region: string | null
  region_rating: number | null
  employment_type: string | null
  employment_tenure_years: number | null
  annual_income: number | null
  monthly_income: number | null
  existing_emis: number | null
  dti: number | null
  credit_score: number | null
  credit_score_display: number | null
  credit_score_basis: string | null
  delinquencies: number | null
  active_loans: number | null
  closed_loans: number | null
  total_loans: number | null
  credit_utilisation: number | null
}

export interface ShortLoanApplicationFormData {
  applicantId?: string
  customer_id: string

  // Section A · Loan request
  loan_amount: number
  loan_purpose: LoanPurposeShort
  loan_tenure_months: number
  repayment_source: RepaymentSource
  preferred_emi_date: number
  end_use_declaration: boolean

  // Section B · What the bank cannot see
  obligations_other_banks: number
  employment_changed_recently: boolean
  additional_income: number
  collateral_type: CollateralType
  collateral_value: number
  is_related_party: boolean

  // Section C · Guarantor (conditional; `guarantor_required` is server-derived)
  guarantor_relationship?: GuarantorRelationship
  guarantor_customer_id?: string
}
