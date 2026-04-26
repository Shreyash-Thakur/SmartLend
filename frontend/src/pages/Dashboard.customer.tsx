import React, { useEffect, useMemo, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { ArrowUpRight, CheckCircle2, Clock3, LogOut, PlusCircle, XCircle, AlertTriangle } from 'lucide-react'
import { DashboardLayout } from '@/components/layouts/DashboardLayout'
import { Button, Card } from '@/components/common'
import { ApplicationTable } from '@/components/sections'
import { useApplicationData } from '@/hooks/useApplicationData'
import { useAuth } from '@/hooks/useAuth'
import type { LoanApplication } from '@/types/application'

export const CustomerDashboard: React.FC = () => {
  const navigate = useNavigate()
  const location = useLocation()
  const { logout, user } = useAuth()
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

  const handleLogout = async () => {
    await logout()
    navigate('/auth')
  }

  return (
    <DashboardLayout title="Customer Dashboard" role="customer">
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

      <div className="flex justify-end mb-6">
        <button
          onClick={() => void handleLogout()}
          className="flex items-center gap-2 px-4 py-2 text-gray-600 hover:text-gray-900 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
        >
          <LogOut className="h-4 w-4" />
          Logout
        </button>
      </div>

      <section className="mb-8">
        <div className="rounded-[36px] bg-gradient-to-br from-[#d9efe5] via-[#edf8f1] to-[#f8fbfe] p-8 shadow-[0_25px_80px_rgba(120,181,166,0.24)]">
          <p className="text-xs uppercase tracking-[0.28em] text-neutral-500">Welcome to SmartLend</p>
          <h2 className="mt-3 text-4xl font-semibold tracking-tight text-neutral-900">Manage Your Loan Application</h2>
          <p className="mt-4 max-w-2xl text-base leading-7 text-neutral-600">
            Submit a new loan application or track the status of your existing applications.
          </p>
        </div>
      </section>

      <section className="mb-8 grid gap-4 md:grid-cols-2">
        <button type="button" className="text-left" onClick={() => navigate('/dashboard/customer/new')}>
          <Card className="rounded-[20px] border-2 border-dashed border-blue-300 bg-blue-50/50 hover:shadow-lg transition-shadow">
            <div className="flex flex-col items-center justify-center py-12 text-center">
              <div className="mb-4 rounded-full bg-blue-100 p-4 text-blue-600"><PlusCircle className="h-8 w-8" /></div>
              <h3 className="text-xl font-semibold text-neutral-900">Start New Application</h3>
              <p className="mt-2 text-sm text-neutral-600">Apply for a new loan with our simple form</p>
              <span className="mt-4 inline-flex rounded-lg bg-blue-600 px-4 py-2 font-semibold text-white">Begin Application</span>
            </div>
          </Card>
        </button>
        <button type="button" className="text-left" onClick={() => navigate('/dashboard/customer?view=history')}>
          <Card className="rounded-[20px] border-2 border-dashed border-indigo-300 bg-indigo-50/50 hover:shadow-lg transition-shadow">
            <div className="flex flex-col items-center justify-center py-12 text-center">
              <div className="mb-4 rounded-full bg-indigo-100 p-4 text-indigo-600"><ArrowUpRight className="h-8 w-8" /></div>
              <h3 className="text-xl font-semibold text-neutral-900">Track Existing Applications</h3>
              <p className="mt-2 text-sm text-neutral-600">View status and details of your submissions</p>
              <span className="mt-4 inline-flex rounded-lg bg-indigo-600 px-4 py-2 font-semibold text-white">View History</span>
            </div>
          </Card>
        </button>
      </section>

      <section className="mb-8">
        <Card title="Application Summary" description="Live status of all your loan applications.">
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
            <div className="rounded-2xl bg-neutral-50 p-4">
              <p className="text-xs uppercase tracking-[0.18em] text-neutral-500">Total</p>
              <p className="mt-2 text-2xl font-semibold text-neutral-900">{snapshot.total}</p>
              <p className="mt-1 text-sm text-neutral-600">Applications submitted</p>
            </div>
            <div className="rounded-2xl bg-green-50 p-4">
              <p className="text-xs uppercase tracking-[0.18em] text-green-800">Approved</p>
              <p className="mt-2 text-2xl font-semibold text-green-900">{snapshot.approved}</p>
              <p className="mt-1 text-sm text-green-800">Confirmed by bank</p>
            </div>
            <div className="rounded-2xl bg-red-50 p-4">
              <p className="text-xs uppercase tracking-[0.18em] text-red-800">Rejected</p>
              <p className="mt-2 text-2xl font-semibold text-red-900">{snapshot.rejected}</p>
              <p className="mt-1 text-sm text-red-800">Not approved</p>
            </div>
            <div className="rounded-2xl bg-amber-50 p-4">
              <p className="text-xs uppercase tracking-[0.18em] text-amber-800">Under Review</p>
              <p className="mt-2 text-2xl font-semibold text-amber-900">{snapshot.pending}</p>
              <p className="mt-1 text-sm text-amber-800">Awaiting bank decision</p>
            </div>
          </div>
          <div className="mt-5 rounded-2xl border border-neutral-200 bg-white p-4 text-sm text-neutral-700">
            SmartLend's hybrid AI evaluates your application instantly. For borderline cases, a bank analyst reviews and sends you their decision with a personal note.
          </div>
        </Card>
      </section>

      {error && (
        <section className="mb-8">
          <Card className="border-red-200 bg-red-50"><p className="text-red-700">{error}</p></Card>
        </section>
      )}

      {(showApplicationHistory || applications.length > 0) && (
        <section>
          <div className="mb-6">
            <h3 className="text-2xl font-semibold text-neutral-900">Your Applications</h3>
            <p className="mt-1 text-neutral-600">{applications.length} application{applications.length !== 1 ? 's' : ''}</p>
          </div>
          {applications.length === 0 ? (
            <Card className="rounded-[36px] border-dashed border-neutral-300 bg-white/75">
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
          <Card className="rounded-[28px] border-amber-200 bg-amber-50">
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
