import React, { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { ArrowUpRight, Clock3, LogOut, PlusCircle } from 'lucide-react'
import { DashboardLayout } from '@/components/layouts/DashboardLayout'
import { Button, Card } from '@/components/common'
import { LoanApplicationForm } from '@/components/forms'
import { ApplicationTable } from '@/components/sections'
import { useApplicationData } from '@/hooks/useApplicationData'
import { useAuth } from '@/hooks/useAuth'
import { trackEvent } from '@/services/analytics'
import type { LoanApplication } from '@/types/application'

export const CustomerDashboard: React.FC = () => {
  const navigate = useNavigate()
  const { logout } = useAuth()
  const [showApplicationForm, setShowApplicationForm] = useState(false)
  const [showApplicationHistory, setShowApplicationHistory] = useState(false)
  const { applications, addApplication, isLoading, error } = useApplicationData({ scope: 'customer' })

  const customerApplications = useMemo(
    () =>
      applications.map((application): LoanApplication => ({
        ...application,
        status: application.status === 'draft' ? application.status : 'submitted',
        finalDecision: undefined,
        decision: application.decision
          ? {
              ...application.decision,
              status: 'deferred' as const,
              explanation: 'Application submitted and waiting for organization review.',
            }
          : application.decision,
      })),
    [applications],
  )

  const handleSubmitApplication = async (data: Parameters<typeof addApplication>[0]) => {
    const application = await addApplication(data)
    trackEvent('application_submitted', { applicationId: application.id })
    setShowApplicationForm(false)
    setShowApplicationHistory(true)
  }

  const handleLogout = async () => {
    await logout()
    navigate('/auth')
  }

  return (
    <DashboardLayout title="Customer Dashboard" role="customer">
      <div className="flex justify-end mb-6">
        <button
          onClick={handleLogout}
          className="flex items-center gap-2 px-4 py-2 text-gray-600 hover:text-gray-900 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
        >
          <LogOut className="h-4 w-4" />
          Logout
        </button>
      </div>

      <section className="mb-8">
        <div className="rounded-[36px] bg-gradient-to-br from-[#d9efe5] via-[#edf8f1] to-[#f8fbfe] p-8 shadow-[0_25px_80px_rgba(120,181,166,0.24)]">
          <div className="flex flex-col items-start justify-between gap-6">
            <div>
              <p className="text-xs uppercase tracking-[0.28em] text-neutral-500">Welcome to SmartLend</p>
              <h2 className="mt-3 text-4xl font-semibold tracking-tight text-neutral-900">
                Manage Your Loan Application
              </h2>
              <p className="mt-4 max-w-2xl text-base leading-7 text-neutral-600">
                Submit a new loan application or track the status of your existing applications.
              </p>
            </div>
          </div>
        </div>
      </section>

      <section className="mb-8 grid gap-4 md:grid-cols-2">
        <button
          type="button"
          className="text-left"
          onClick={() => setShowApplicationForm(true)}
        >
          <Card className="rounded-[20px] border-2 border-dashed border-blue-300 bg-blue-50/50 hover:shadow-lg transition-shadow">
            <div className="flex flex-col items-center justify-center py-12 text-center">
              <div className="mb-4 rounded-full bg-blue-100 p-4 text-blue-600">
                <PlusCircle className="h-8 w-8" />
              </div>
              <h3 className="text-xl font-semibold text-neutral-900">Start New Application</h3>
              <p className="mt-2 text-sm text-neutral-600">Apply for a new loan with our simple form</p>
              <span className="mt-4 inline-flex rounded-lg bg-blue-600 px-4 py-2 font-semibold text-white">
                Begin Application
              </span>
            </div>
          </Card>
        </button>

        <button
          type="button"
          className="text-left"
          onClick={() => setShowApplicationHistory(true)}
        >
          <Card className="rounded-[20px] border-2 border-dashed border-indigo-300 bg-indigo-50/50 hover:shadow-lg transition-shadow">
            <div className="flex flex-col items-center justify-center py-12 text-center">
              <div className="mb-4 rounded-full bg-indigo-100 p-4 text-indigo-600">
                <ArrowUpRight className="h-8 w-8" />
              </div>
              <h3 className="text-xl font-semibold text-neutral-900">Track Existing Applications</h3>
              <p className="mt-2 text-sm text-neutral-600">View status and details of your submissions</p>
              <span className="mt-4 inline-flex rounded-lg bg-indigo-600 px-4 py-2 font-semibold text-white">
                View History
              </span>
            </div>
          </Card>
        </button>
      </section>

      {error && (
        <section className="mb-8">
          <Card className="border-red-200 bg-red-50">
            <p className="text-red-700">{error}</p>
          </Card>
        </section>
      )}

      {showApplicationForm && (
        <section className="mb-8">
          <Card
            title="New Loan Application"
            description="Fill in your details to apply for a loan. All fields are required."
            className="rounded-[36px] border-white/80"
          >
            <LoanApplicationForm
              onSubmit={handleSubmitApplication}
              isLoading={isLoading}
              isMultiStep
            />
          </Card>
        </section>
      )}

      {(showApplicationHistory || applications.length > 0) && (
        <section>
          <div className="mb-6 flex items-center justify-between">
            <div>
              <h3 className="text-2xl font-semibold text-neutral-900">Your Applications</h3>
              <p className="mt-1 text-neutral-600">
                You have {applications.length} application{applications.length !== 1 ? 's' : ''}
              </p>
            </div>
          </div>
          {applications.length === 0 ? (
            <Card className="rounded-[36px] border-dashed border-neutral-300 bg-white/75">
              <div className="flex flex-col items-center justify-center py-16 text-center">
                <div className="mb-4 rounded-full bg-primary-50 p-4 text-primary-600">
                  <PlusCircle className="h-8 w-8" />
                </div>
                <h3 className="text-2xl font-semibold text-neutral-900">No applications submitted yet</h3>
                <p className="mt-3 max-w-lg text-neutral-600">Submit your first application to begin organization review.</p>
                <Button
                  className="mt-6 rounded-2xl"
                  rightIcon={<ArrowUpRight className="h-4 w-4" />}
                  onClick={() => setShowApplicationForm(true)}
                >
                  Create First Application
                </Button>
              </div>
            </Card>
          ) : (
            <ApplicationTable data={customerApplications} isLoading={isLoading} />
          )}
        </section>
      )}

      {applications.length > 0 && (
        <section className="mt-8">
          <Card className="rounded-[28px] border-amber-200 bg-amber-50">
            <div className="flex items-start gap-3">
              <Clock3 className="mt-1 h-5 w-5 text-amber-700" />
              <div>
                <h3 className="text-lg font-semibold text-amber-950">Under Review</h3>
                <p className="mt-1 text-sm text-amber-900">
                  Submitted applications are shown as under review in the customer dashboard. Final decisions are completed by the organization review team.
                </p>
              </div>
            </div>
          </Card>
        </section>
      )}
    </DashboardLayout>
  )
}
