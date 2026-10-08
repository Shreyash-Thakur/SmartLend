import React, { useEffect, useMemo, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { ArrowRight, ArrowUpRight, CheckCircle2, Clock3, FileText, History, PlusCircle, XCircle, AlertTriangle } from 'lucide-react'
import { DashboardLayout } from '@/components/layouts/DashboardLayout'
import { Button, Card, KPICard } from '@/components/common'
import { ApplicationTable } from '@/components/sections'
import { useApplicationData } from '@/hooks/useApplicationData'
import { useAuth } from '@/hooks/useAuth'
import type { LoanApplication } from '@/types/application'

export const CustomerDashboard: React.FC = () => {
  const navigate = useNavigate()
  const location = useLocation()
  const { user } = useAuth()
  const [showApplicationHistory, setShowApplicationHistory] = useState(() =>
    new URLSearchParams(location.search).get('view') === 'history',
  )
  const { applications, isLoading, error } = useApplicationData({ scope: 'customer', applicantId: user?.uid })

  const customerApplications = useMemo(
    () => applications.map((application): LoanApplication => ({ ...application })),
    [applications],
  )

  useEffect(() => {
    setShowApplicationHistory(new URLSearchParams(location.search).get('view') === 'history')
  }, [location.search])

  const snapshot = useMemo(() => {
    const total = applications.length
    const pending = applications.filter((item) => item.status === 'deferred').length
    const approved = applications.filter((item) => item.status === 'approved').length
    const rejected = applications.filter((item) => item.status === 'rejected').length
    return { total, pending, approved, rejected }
  }, [applications])

  const firstName = user?.displayName?.split(' ')[0]

  return (
    <DashboardLayout title="Dashboard" role="customer">
      {/* Page header */}
      <section className="mb-6 flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight text-neutral-900 sm:text-3xl">
            {firstName ? `Welcome, ${firstName}` : 'Welcome to SmartLend'}
          </h1>
          <p className="mt-1 text-sm text-neutral-500">Submit a new loan application or track the status of your existing applications.</p>
        </div>
        <Button size="sm" leftIcon={<PlusCircle className="h-4 w-4" />} onClick={() => navigate('/dashboard/customer/new')}>
          New Application
        </Button>
      </section>

      {/* Deferred notification */}
      {snapshot.pending > 0 && (
        <section className="mb-6">
          <div className="rounded-2xl p-4 flex items-center gap-3 bg-amber-50 border border-amber-200">
            <AlertTriangle className="h-5 w-5 text-amber-600 shrink-0" />
            <p className="text-sm text-amber-800">
              <span className="font-semibold">{snapshot.pending} application{snapshot.pending > 1 ? 's are' : ' is'} under manual review.</span>{' '}
              The bank team is evaluating your case. You'll see their decision and notes here once confirmed.
            </p>
          </div>
        </section>
      )}

      {/* Summary */}
      <section className="mb-6 grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <KPICard label="Total" value={snapshot.total} icon={FileText} tone="blue" hint="Applications submitted" />
        <KPICard label="Approved" value={snapshot.approved} icon={CheckCircle2} tone="green" hint="Confirmed by bank" />
        <KPICard label="Rejected" value={snapshot.rejected} icon={XCircle} tone="red" hint="Not approved" />
        <KPICard label="Under Review" value={snapshot.pending} icon={Clock3} tone="amber" hint="Awaiting bank decision" />
      </section>

      {/* Actions */}
      <section className="mb-8 grid gap-4 md:grid-cols-2">
        <ActionCard
          icon={<PlusCircle className="h-6 w-6" />}
          tone="bg-blue-50 text-blue-600"
          title="Start New Application"
          text="Apply for a new loan with our simple form."
          onClick={() => navigate('/dashboard/customer/new')}
        />
        <ActionCard
          icon={<History className="h-6 w-6" />}
          tone="bg-violet-50 text-violet-600"
          title="Track Existing Applications"
          text="View status and details of your submissions."
          onClick={() => navigate('/dashboard/customer?view=history')}
        />
      </section>

      {error && (
        <section className="mb-8">
          <Card className="border-red-200 bg-red-50"><p className="text-red-700">{error}</p></Card>
        </section>
      )}

      {(showApplicationHistory || applications.length > 0) && (
        <section>
          <div className="mb-4">
            <h3 className="text-lg font-semibold text-neutral-900">Your Applications</h3>
            <p className="mt-1 text-neutral-600">{applications.length} application{applications.length !== 1 ? 's' : ''}</p>
          </div>
          {applications.length === 0 ? (
            <Card className="border-dashed border-neutral-300">
              <div className="flex flex-col items-center justify-center py-16 text-center">
                <div className="mb-4 rounded-full bg-primary-50 p-4 text-primary-600"><PlusCircle className="h-8 w-8" /></div>
                <h3 className="text-2xl font-semibold text-neutral-900">No applications yet</h3>
                <p className="mt-3 max-w-lg text-neutral-600">Submit your first application to begin review.</p>
                <Button className="mt-6 rounded-2xl" rightIcon={<ArrowUpRight className="h-4 w-4" />} onClick={() => navigate('/dashboard/customer/new')}>
                  Create First Application
                </Button>
              </div>
            </Card>
          ) : (
            <ApplicationTable data={customerApplications} isLoading={isLoading} onRowClick={(app) => navigate(`/review/${app.id}`)} hideMetrics={true} />
          )}
        </section>
      )}

      {applications.length > 0 && (
        <section className="mt-8">
          <Card className="border-amber-200 bg-amber-50">
            <div className="flex items-start gap-3">
              <Clock3 className="mt-1 h-5 w-5 text-amber-700" />
              <div>
                <h3 className="text-lg font-semibold text-amber-950">How decisions work</h3>
                <p className="mt-1 text-sm text-amber-900">
                  Our AI provides an instant recommendation. Deferred cases go to a bank analyst who reviews the details and sends you their decision with a personal note.
                </p>
              </div>
            </div>
          </Card>
        </section>
      )}
    </DashboardLayout>
  )
}

function ActionCard({ icon, tone, title, text, onClick }: {
  icon: React.ReactNode
  tone: string
  title: string
  text: string
  onClick: () => void
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="group flex items-center gap-4 rounded-2xl border border-neutral-200 bg-white p-5 text-left shadow-[0_1px_2px_rgba(15,23,42,0.04)] transition-all hover:-translate-y-0.5 hover:border-neutral-300 hover:shadow-[0_8px_30px_rgba(15,23,42,0.06)]"
    >
      <span className={`flex h-12 w-12 shrink-0 items-center justify-center rounded-xl ${tone}`}>{icon}</span>
      <span className="min-w-0 flex-1">
        <span className="block font-semibold text-neutral-900">{title}</span>
        <span className="mt-0.5 block text-sm text-neutral-500">{text}</span>
      </span>
      <ArrowRight className="h-5 w-5 shrink-0 text-neutral-400 transition-transform group-hover:translate-x-0.5 group-hover:text-neutral-900" />
    </button>
  )
}
