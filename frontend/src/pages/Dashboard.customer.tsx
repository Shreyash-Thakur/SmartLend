import React from 'react'
import { useNavigate } from 'react-router-dom'
import { ArrowUpRight, PlusCircle, ShieldCheck, Sparkles } from 'lucide-react'
import { DashboardLayout } from '@/components/layouts/DashboardLayout'
import { Button, Card } from '@/components/common'
import { LoanApplicationForm } from '@/components/forms'
import { ApplicationTable } from '@/components/sections'
import { useApplicationData } from '@/hooks/useApplicationData'
import { formatCurrency } from '@/lib/utils'
import { trackEvent } from '@/services/analytics'

export const CustomerDashboard: React.FC = () => {
  const navigate = useNavigate()
  const { applications, addApplication, isLoading } = useApplicationData({ scope: 'customer' })

  const handleSubmitApplication = async (data: Parameters<typeof addApplication>[0]) => {
    const application = await addApplication(data)
    trackEvent('application_submitted', { applicationId: application.id })
    navigate(`/review/${application.id}`)
  }

  const handleRowClick = (application: { id: string }) => navigate(`/review/${application.id}`)

  const approvedCount = applications.filter((a) => a.status === 'approved').length
  const reviewCount = applications.filter((a) => ['deferred', 'processing'].includes(a.status)).length
  const submittedValue = applications.reduce((sum, application) => sum + application.loanAmount, 0)

  return (
    <DashboardLayout title="Loan Application" role="customer">
      <section className="mb-8 grid gap-6 lg:grid-cols-[1.35fr_0.65fr]">
        <div className="rounded-[36px] bg-gradient-to-br from-[#d9efe5] via-[#edf8f1] to-[#f8fbfe] p-8 shadow-[0_25px_80px_rgba(120,181,166,0.24)]">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div className="max-w-2xl">
              <p className="text-xs uppercase tracking-[0.28em] text-neutral-500">Customer Workspace</p>
              <h2 className="mt-3 text-4xl font-semibold tracking-tight text-neutral-900">
                Start a new SmartLend application
              </h2>
              <p className="mt-4 max-w-xl text-base leading-7 text-neutral-600">
                Your dashboard now shows only applications created in this session. Organization seed
                records stay on the org side, so this space feels like a real applicant portal.
              </p>
            </div>
            <div className="grid gap-3 sm:grid-cols-2">
              <QuickChip icon={<Sparkles className="h-4 w-4" />} label="Session-only history" />
              <QuickChip icon={<ShieldCheck className="h-4 w-4" />} label="33-column intake ready" />
            </div>
          </div>
        </div>

        <div className="grid gap-4">
          <StatPanel label="Applications Created" value={String(applications.length)} accent="from-neutral-900 to-neutral-700" />
          <StatPanel label="Approved" value={String(approvedCount)} accent="from-emerald-500 to-green-600" />
          <StatPanel label="Under Review" value={String(reviewCount)} accent="from-amber-400 to-orange-500" />
          <StatPanel label="Requested Value" value={formatCurrency(submittedValue)} accent="from-sky-500 to-cyan-500" />
        </div>
      </section>

      <div className="grid grid-cols-1 gap-8 lg:grid-cols-3">
        <section className="lg:col-span-2">
          <Card
            title="New Loan Application"
            description="Capture applicant identity, income, assets, credit history, region, city, and derived underwriting ratios."
            className="rounded-[36px] border-white/80"
          >
            <LoanApplicationForm
              onSubmit={handleSubmitApplication}
              isLoading={isLoading}
              isMultiStep
            />
          </Card>
        </section>

        <aside className="lg:col-span-1 space-y-6">
          <Card title="Session Summary" className="rounded-[32px]">
            <div className="space-y-4">
              <div>
                <p className="text-sm text-neutral-600">Total Applications</p>
                <p className="text-3xl font-bold text-neutral-900">{applications.length}</p>
              </div>
              <div className="border-t border-neutral-200 pt-4">
                <p className="text-sm text-neutral-600">Approved</p>
                <p className="text-2xl font-bold text-green-600">
                  {approvedCount}
                </p>
              </div>
              <div className="border-t border-neutral-200 pt-4">
                <p className="text-sm text-neutral-600">Pending Review</p>
                <p className="text-2xl font-bold text-deferred">{reviewCount}</p>
              </div>
              <div className="border-t border-neutral-200 pt-4">
                <p className="text-sm text-neutral-600">Requested Value</p>
                <p className="text-2xl font-bold text-neutral-900">{formatCurrency(submittedValue)}</p>
              </div>
            </div>
          </Card>

          <Card title="Applicant Guidance" className="rounded-[32px]">
            <ul className="space-y-3 text-sm text-neutral-700">
              <li className="flex gap-2">
                <span className="text-primary-500">✓</span>
                <span>Provide exact asset splits and current liabilities</span>
              </li>
              <li className="flex gap-2">
                <span className="text-primary-500">✓</span>
                <span>Upload bank or bureau documents for faster analyst review</span>
              </li>
              <li className="flex gap-2">
                <span className="text-primary-500">✓</span>
                <span>Higher CIBIL and lower EMI ratios improve confidence</span>
              </li>
              <li className="flex gap-2">
                <span className="text-primary-500">✓</span>
                <span>Only applications created in this session appear below</span>
              </li>
            </ul>
          </Card>
        </aside>
      </div>

      <section className="mt-8">
        {applications.length === 0 ? (
          <Card className="rounded-[36px] border-dashed border-neutral-300 bg-white/75">
            <div className="flex flex-col items-center justify-center py-16 text-center">
              <div className="mb-4 rounded-full bg-primary-50 p-4 text-primary-600">
                <PlusCircle className="h-8 w-8" />
              </div>
              <h3 className="text-2xl font-semibold text-neutral-900">No customer applications yet</h3>
              <p className="mt-3 max-w-lg text-neutral-600">
                This dashboard starts empty by design. Once you create an application in this session,
                it will appear here and remain separate from the organization sample records.
              </p>
              <Button
                className="mt-6 rounded-2xl"
                rightIcon={<ArrowUpRight className="h-4 w-4" />}
                onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
              >
                Create First Application
              </Button>
            </div>
          </Card>
        ) : (
          <ApplicationTable data={applications} onRowClick={handleRowClick} isLoading={isLoading} />
        )}
      </section>
    </DashboardLayout>
  )
}

function QuickChip({ icon, label }: { icon: React.ReactNode; label: string }) {
  return (
    <div className="inline-flex items-center gap-2 rounded-full bg-white/80 px-4 py-2 text-sm font-medium text-neutral-700 shadow-sm">
      {icon}
      {label}
    </div>
  )
}

function StatPanel({ label, value, accent }: { label: string; value: string; accent: string }) {
  return (
    <div className="rounded-[28px] border border-white/70 bg-white/85 p-5 shadow-lg backdrop-blur-xl">
      <div className={`h-1.5 w-20 rounded-full bg-gradient-to-r ${accent}`} />
      <p className="mt-4 text-sm text-neutral-500">{label}</p>
      <p className="mt-2 text-3xl font-semibold text-neutral-900">{value}</p>
    </div>
  )
}
