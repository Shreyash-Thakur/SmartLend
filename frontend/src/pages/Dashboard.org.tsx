import React, { useEffect, useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import { Activity, BriefcaseBusiness, Clock3, MapPinned, Users } from 'lucide-react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { DashboardLayout } from '@/components/layouts/DashboardLayout'
import { Button, Card, KPICard } from '@/components/common'
import { ApplicationTable } from '@/components/sections'
import { useApplicationData } from '@/hooks/useApplicationData'
import { useUiStore } from '@/store/uiStore'
import { getStats } from '@/services/applications'
import type { StatsResponse } from '@/types/api'
import { formatCurrency } from '@/lib/utils'

export const OrganizationDashboard: React.FC = () => {
  const navigate = useNavigate()
  const { applications, isLoading, error } = useApplicationData({ scope: 'org' })
  const { activeTab, setActiveTab } = useUiStore()
  const [stats, setStats] = React.useState<StatsResponse | null>(null)
  const [dashboardError, setDashboardError] = React.useState<string | null>(null)
  const [statsLoading, setStatsLoading] = React.useState(true)

  useEffect(() => {
    const loadDashboardData = async () => {
      setDashboardError(null)
      setStatsLoading(true)
      try {
        const statsResponse = await getStats()
        setStats(statsResponse)
      } catch (fetchError) {
        setDashboardError(fetchError instanceof Error ? fetchError.message : 'Failed to load dashboard data')
      } finally {
        setStatsLoading(false)
      }
    }

    void loadDashboardData()
  }, [])

  const trends = useMemo(() => {
    const now = new Date()
    const weekWindows = [
      { label: 'Week 1', start: new Date(now.getTime() - 28 * 24 * 60 * 60 * 1000), end: new Date(now.getTime() - 21 * 24 * 60 * 60 * 1000) },
      { label: 'Week 2', start: new Date(now.getTime() - 21 * 24 * 60 * 60 * 1000), end: new Date(now.getTime() - 14 * 24 * 60 * 60 * 1000) },
      { label: 'Week 3', start: new Date(now.getTime() - 14 * 24 * 60 * 60 * 1000), end: new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000) },
      { label: 'Week 4', start: new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000), end: new Date(now.getTime() + 24 * 60 * 60 * 1000) },
    ]

    return weekWindows.map((window) => {
      const bucket = applications.filter((application) => {
        const created = new Date(application.createdAt)
        return created >= window.start && created < window.end
      })

      return {
        date: window.label,
        count: bucket.length,
        approved: bucket.filter((application) => application.finalDecision === 'APPROVE').length,
        rejected: bucket.filter((application) => application.finalDecision === 'REJECT').length,
        deferred: bucket.filter((application) => application.finalDecision === 'DEFER').length,
      }
    })
  }, [applications])

  const approvalDistribution = useMemo(
    () => [
      { label: 'Approved', value: stats?.approved ?? 0, fill: '#10b981' },
      { label: 'Rejected', value: stats?.rejected ?? 0, fill: '#ef4444' },
      { label: 'Deferred', value: stats?.deferred ?? 0, fill: '#ec4899' },
      {
        label: 'Processing',
        value: applications.filter((application) => application.status === 'processing').length,
        fill: '#f59e0b',
      },
    ],
    [applications, stats],
  )

  const categoryAnalysis = useMemo(() => {
    const counts = new Map<string, number>()
    for (const application of applications) {
      counts.set(application.loanPurpose, (counts.get(application.loanPurpose) ?? 0) + 1)
    }
    return Array.from(counts.entries()).map(([label, value]) => ({
      label: `${label.charAt(0).toUpperCase()}${label.slice(1)} Loan`,
      value,
    }))
  }, [applications])

  const riskScoreDistribution = useMemo(() => {
    const buckets = { low: 0, medium: 0, high: 0 }
    for (const application of applications) {
      const risk = application.decision?.riskScore ?? 0
      if (risk < 0.3) buckets.low += 1
      else if (risk < 0.6) buckets.medium += 1
      else buckets.high += 1
    }
    return [
      { label: 'Low Risk', value: buckets.low, fill: '#10b981' },
      { label: 'Medium Risk', value: buckets.medium, fill: '#f59e0b' },
      { label: 'High Risk', value: buckets.high, fill: '#ef4444' },
    ]
  }, [applications])

  const displayApplications =
    activeTab === 'deferred'
      ? applications.filter((a) => a.status === 'deferred' || a.status === 'submitted')
      : applications
  const uploadedCount = applications.filter((a) => a.source === 'seed').length
  const submittedCount = applications.filter((a) => a.source === 'customer').length

  const handleRowClick = (application: { id: string }) => navigate(`/review/${application.id}`)

  return (
    <DashboardLayout title="Organization Dashboard" role="organization">
      <section className="mb-8 grid gap-6 lg:grid-cols-[1.25fr_0.75fr]">
        <div className="rounded-[36px] border border-[#d6e7e4] bg-gradient-to-br from-[#edf6f4] via-[#f6faf9] to-[#f9fcfd] p-8 shadow-[0_30px_100px_rgba(118,176,165,0.18)]">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div className="max-w-2xl">
              <p className="text-xs uppercase tracking-[0.28em] text-neutral-500">Operations Dashboard</p>
              <h2 className="mt-3 text-4xl font-semibold tracking-tight text-neutral-900">
                Unified application pipeline
              </h2>
              <p className="mt-4 text-base leading-7 text-neutral-600">
                All records below come from the live backend and are shown in one operational queue.
                Applications are grouped by source as Uploaded and Submitted for faster triage.
              </p>
            </div>
            <div className="grid gap-3 sm:grid-cols-2">
              <OrgChip icon={<Users className="h-4 w-4" />} label={`${uploadedCount} uploaded applications`} />
              <OrgChip icon={<BriefcaseBusiness className="h-4 w-4" />} label={`${submittedCount} submitted applications`} />
              <OrgChip icon={<Activity className="h-4 w-4" />} label="Analyst workflow active" />
              <OrgChip icon={<Clock3 className="h-4 w-4" />} label="Real-time backend sync" />
            </div>
          </div>
          <div className="mt-6">
            <div className="flex flex-wrap gap-3">
              <Button variant="secondary" onClick={() => navigate('/dashboard/models')}>
                Open Model Analysis Dashboard
              </Button>
              <Button variant="primary" leftIcon={<MapPinned className="h-4 w-4" />} onClick={() => navigate('/analytics/geo')}>
                Open Geo Analytics
              </Button>
            </div>
          </div>
        </div>

        <div className="grid gap-4">
          <Card className="rounded-[30px] border-white/80 bg-neutral-900 text-white">
            <div className="space-y-3">
              <p className="text-xs uppercase tracking-[0.2em] text-white/60">Application Sources</p>
              <p className="text-4xl font-semibold">{applications.length}</p>
              <div className="grid gap-3 text-sm text-white/75">
                <div className="flex items-center justify-between">
                  <span>Uploaded</span>
                  <span>{uploadedCount}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span>Submitted</span>
                  <span>{submittedCount}</span>
                </div>
              </div>
            </div>
          </Card>
        </div>
      </section>

      {(error || dashboardError) && (
        <section className="mb-8">
          <Card className="border-red-200 bg-red-50">
            <p className="text-red-700">
              Connection issue: {error ?? dashboardError}
            </p>
          </Card>
        </section>
      )}

      <section className="mb-8">
        <Card title="Live Stats">
          {statsLoading ? (
            <p className="text-neutral-600">Loading stats...</p>
          ) : stats ? (
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
              <KPICard label="Total Applications" value={stats.totalApplications} format="number" />
              <KPICard label="Approval Rate" value={stats.approvalRate} format="percentage" />
              <KPICard label="Rejection Rate" value={stats.rejectionRate} format="percentage" />
              <KPICard label="Deferral Rate" value={stats.deferralRate} format="percentage" />
              <KPICard label="Average CBES" value={stats.averageCBES} format="number" />
              <KPICard label="Average ML Score" value={stats.averageMLProbability} format="number" />
            </div>
          ) : (
            <p className="text-neutral-600">Stats unavailable</p>
          )}
        </Card>
      </section>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        <Card title="Approval Distribution">
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={approvalDistribution} dataKey="value" nameKey="label" outerRadius={100}>
                  {approvalDistribution.map((entry) => (
                    <Cell key={entry.label} fill={entry.fill} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card title="Applications Over Time">
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={trends} margin={{ top: 16, right: 24, bottom: 8, left: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="date" />
                <YAxis allowDecimals={false} width={42} />
                <Tooltip />
                <Line type="monotone" dataKey="count" stroke="#16a34a" strokeWidth={3} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card title="Category Analysis">
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={categoryAnalysis} margin={{ top: 16, right: 24, bottom: 36, left: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="label" interval={0} angle={-18} textAnchor="end" height={60} />
                <YAxis allowDecimals={false} width={42} />
                <Tooltip />
                <Bar dataKey="value" fill="#0ea5e9" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card title="Risk Score Distribution">
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={riskScoreDistribution} margin={{ top: 16, right: 24, bottom: 8, left: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="label" />
                <YAxis allowDecimals={false} width={42} />
                <Tooltip />
                <Bar dataKey="value" radius={[8, 8, 0, 0]}>
                  {riskScoreDistribution.map((entry) => (
                    <Cell key={entry.label} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      <div className="mb-6 rounded-xl border border-neutral-200 bg-white p-4 shadow-md">
        <div className="grid gap-4 md:grid-cols-3">
          <h3 className="text-lg font-semibold text-neutral-900 mb-6">Operations Snapshot</h3>
          <div>
            <p className="text-sm text-neutral-500">Average Loan Amount</p>
            <p className="text-2xl font-semibold text-neutral-900">
              {stats ? formatCurrency(Math.round(applications.reduce((sum, app) => sum + app.loanAmount, 0) / Math.max(applications.length, 1))) : '--'}
            </p>
          </div>
          <div>
            <p className="text-sm text-neutral-500">Source Mix</p>
            <p className="text-2xl font-semibold text-neutral-900">
              {applications.length ? `${Math.round((submittedCount / applications.length) * 100)}% submitted` : '--'}
            </p>
          </div>
          <div>
            <p className="text-sm text-neutral-500">Human Review Queue</p>
            <p className="text-2xl font-semibold text-neutral-900">
              {applications.filter((application) => application.status === 'deferred' || application.status === 'submitted').length}
            </p>
          </div>
        </div>
      </div>

      <div className="bg-white rounded-xl border border-neutral-200 shadow-md overflow-hidden">
        <div className="border-b border-neutral-200 flex">
          <button
            onClick={() => setActiveTab('all')}
            className={`flex-1 px-6 py-4 text-center font-medium transition-colors ${
              activeTab === 'all'
                ? 'text-primary-600 border-b-2 border-primary-600 bg-primary-50'
                : 'text-neutral-600 hover:text-neutral-900'
            }`}
          >
            All Applications ({applications.length})
          </button>
          <button
            onClick={() => setActiveTab('deferred')}
            className={`flex-1 px-6 py-4 text-center font-medium transition-colors ${
              activeTab === 'deferred'
                ? 'text-primary-600 border-b-2 border-primary-600 bg-primary-50'
                : 'text-neutral-600 hover:text-neutral-900'
            }`}
          >
            Pending Review ({applications.filter((a) => a.status === 'deferred' || a.status === 'submitted').length})
          </button>
        </div>

        <div className="p-6">
          <ApplicationTable
            data={displayApplications}
            onRowClick={handleRowClick}
            isLoading={isLoading}
            pageSize={25}
            showApplicant
          />
        </div>
      </div>
    </DashboardLayout>
  )
}

function OrgChip({ icon, label }: { icon: React.ReactNode; label: string }) {
  return (
    <div className="inline-flex items-center gap-2 rounded-full bg-white/80 px-4 py-2 text-sm font-medium text-neutral-700 shadow-sm">
      {icon}
      {label}
    </div>
  )
}
