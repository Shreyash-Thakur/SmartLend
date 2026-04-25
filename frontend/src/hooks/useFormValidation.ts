import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z, type ZodTypeAny } from 'zod'

export const loanApplicationSchema = z
  .object({
    applicantId: z.string().optional(),
    firstName: z.string().min(2).max(50),
    lastName: z.string().min(2).max(50),
    email: z.string().email(),
    phone: z.string().min(10).max(15),
    gender: z.enum(['male', 'female', 'other']),
    maritalStatus: z.enum(['single', 'married', 'divorced', 'widowed']).optional(),
    education: z.enum(['high_school', 'diploma', 'graduate', 'postgraduate', 'doctorate']).optional(),
    loanAmount: z.number().min(1).max(10000000),
    loanPurpose: z.enum(['home', 'auto', 'personal', 'business', 'education']),
    loanTenure: z.number().min(12).max(360),
    interestRate: z.number().min(1).max(30).optional(),
    monthlyIncome: z.number().min(15000).max(500000),
    annualIncome: z.number().min(180000).max(6000000).optional(),
    emi: z.number().min(0),
    existingEmis: z.number().min(0).optional(),
    assets: z.number().min(0),
    residentialAssetsValue: z.number().min(0).optional(),
    commercialAssetsValue: z.number().min(0).optional(),
    bankBalance: z.number().min(0).optional(),
    totalAssets: z.number().min(0).optional(),
    liabilities: z.number().min(0).optional(),
    creditScore: z.number().min(300).max(900).optional(),
    cibilScore: z.number().min(300).max(900),
    creditHistory: z.enum(['excellent', 'good', 'average', 'poor']).optional(),
    totalLoans: z.number().min(0).max(50).optional(),
    activeLoans: z.number().min(0).max(25).optional(),
    closedLoans: z.number().min(0).max(50).optional(),
    missedPayments: z.number().min(0).max(100).optional(),
    creditUtilizationRatio: z.number().min(0).max(100).optional(),
    emiIncomeRatio: z.number().min(0).max(100).optional(),
    loanIncomeRatio: z.number().min(0).max(1000).optional(),
    debtToIncomeRatio: z.number().min(0).max(100).optional(),
    age: z.number().min(21).max(65),
    dependents: z.number().min(0).max(10).optional(),
    employmentType: z.enum(['salaried', 'self-employed', 'business', 'retired']),
    yearsOfEmployment: z.number().min(0).max(45).optional(),
    residenceType: z.enum(['owned', 'rented', 'with_family']).optional(),
    region: z.enum(['rural', 'urban', 'semi_urban']),
    city: z.string().min(2).max(80).optional(),
  })
  .superRefine((data, ctx) => {
    if (data.emi > data.monthlyIncome * 0.4) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: 'EMI should not exceed 40% of monthly income',
        path: ['emi'],
      })
    }
    if ((data.activeLoans ?? 0) > (data.totalLoans ?? 0)) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: 'Active loans cannot exceed total loans',
        path: ['activeLoans'],
      })
    }
    if ((data.closedLoans ?? 0) > (data.totalLoans ?? 0)) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: 'Closed loans cannot exceed total loans',
        path: ['closedLoans'],
      })
    }
  })

export type LoanApplicationSchema = z.infer<typeof loanApplicationSchema>

export function useFormValidation<TSchema extends ZodTypeAny>(schema: TSchema) {
  return useForm<z.infer<TSchema>>({
    resolver: zodResolver(schema),
    mode: 'onChange',
  })
}
