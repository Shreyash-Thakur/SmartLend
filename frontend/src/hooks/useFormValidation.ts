import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z, type ZodTypeAny } from 'zod'

const emptyToUndefined = (value: unknown) => {
  if (value === '' || value === null || value === undefined) return undefined
  if (typeof value === 'number' && Number.isNaN(value)) return undefined
  return value
}

const optionalEnum = <T extends [string, ...string[]]>(values: T) =>
  z.preprocess(emptyToUndefined, z.enum(values).optional())

const optionalNumber = (schema: z.ZodNumber) => z.preprocess(emptyToUndefined, schema.optional())

export const loanApplicationSchema = z
  .object({
    applicantId: z.string().optional(),
    firstName: z.string().min(2).max(50),
    lastName: z.string().min(2).max(50),
    email: z.string().email(),
    phone: z.string().min(10).max(15),
    gender: z.enum(['male', 'female', 'other']),
    maritalStatus: optionalEnum(['single', 'married', 'divorced', 'widowed'] as const),
    education: optionalEnum(['high_school', 'diploma', 'graduate', 'postgraduate', 'doctorate'] as const),
    loanAmount: z.number().min(1).max(10000000),
    loanPurpose: z.enum(['home', 'auto', 'personal', 'business', 'education']),
    loanTenure: z.number().min(12).max(360),
    interestRate: optionalNumber(z.number().min(1).max(30)),
    monthlyIncome: z.number().min(15000).max(500000),
    annualIncome: optionalNumber(z.number().min(180000).max(6000000)),
    emi: z.number().min(0),
    existingEmis: optionalNumber(z.number().min(0)),
    assets: z.number().min(0),
    residentialAssetsValue: optionalNumber(z.number().min(0)),
    commercialAssetsValue: optionalNumber(z.number().min(0)),
    bankBalance: optionalNumber(z.number().min(0)),
    totalAssets: optionalNumber(z.number().min(0)),
    liabilities: optionalNumber(z.number().min(0)),
    creditScore: optionalNumber(z.number().min(300).max(900)),
    cibilScore: z.number().min(300).max(900),
    creditHistory: optionalEnum(['excellent', 'good', 'average', 'poor'] as const),
    totalLoans: optionalNumber(z.number().min(0).max(50)),
    activeLoans: optionalNumber(z.number().min(0).max(25)),
    closedLoans: optionalNumber(z.number().min(0).max(50)),
    missedPayments: optionalNumber(z.number().min(0).max(100)),
    creditUtilizationRatio: optionalNumber(z.number().min(0).max(100)),
    emiIncomeRatio: optionalNumber(z.number().min(0).max(100)),
    loanIncomeRatio: optionalNumber(z.number().min(0).max(1000)),
    debtToIncomeRatio: optionalNumber(z.number().min(0).max(100)),
    age: z.number().min(21).max(65),
    dependents: optionalNumber(z.number().min(0).max(10)),
    employmentType: z.enum(['salaried', 'self-employed', 'business', 'retired']),
    yearsOfEmployment: optionalNumber(z.number().min(0).max(45)),
    residenceType: optionalEnum(['owned', 'rented', 'with_family'] as const),
    region: z.enum(['rural', 'urban', 'semi_urban']),
    city: z.preprocess(emptyToUndefined, z.string().min(2).max(80).optional()),
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
