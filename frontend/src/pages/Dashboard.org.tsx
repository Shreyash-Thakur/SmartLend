import React, { useEffect, useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import { Activity, BriefcaseBusiness, Clock3, Users } from 'lucide-react'
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
import { Card, KPICard } from '@/components/common'
import { ApplicationTable, MetricsGrid } from '@/components/sections'
import { useApplicationData } from '@/hooks/useApplicationData'
import { useUiStore } from '@/store/uiStore'
import { getDashboardMetrics, getTrendData } from '@/services/applications'
import type { DashboardMetrics, TrendDataPoint } from '@/types/api'
import { formatCurrency } from '@/lib/utils'

export const OrganizationDashboard: React.FC = () => {
  const navigate = useNavigate()
  const { applications } = useApplicationData({ scope: 'org' })
  const { activeTab, setActiveTab } = useUiStore()
  const [metrics, setMetrics] = React.useState<DashboardMetrics | null>(null)
  const [trends, setTrends] = React.useState<TrendDataPoint[]>([])

  useEffect(() => {
    void getDashboardMetrics().then(setMetrics)
    void getTrendData().then(setTrends)
  }, [])

  const approvalDistribution = useMemo(
    () => [
      { label: 'Approved', value: metrics?.approved ?? 0, fill: '#10b981' },
      { label: 'Rejected', value: metrics?.rejected ?? 0, fill: '#ef4444' },
      { label: 'Deferred', value: metrics?.deferred ?? 0, fill: '#ec4899' },
      {
        label: 'Processing',
        value: applications.filter((application) => application.status === 'processing').length,
        fill: '#f59e0b',
      },
    ],
    [applications, metrics],
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
      ? applications.filter((a) => a.status === 'deferred')
      : applications

  const handleRowClick = (application: { id: string }) => navigate(`/review/${application.id}`)

  return (
    <DashboardLayout title="Organization Dashboard" role="organization">
      <section className="mb-8 grid gap-6 lg:grid-cols-[1.25fr_0.75fr]">
        <div className="rounded-[36px] bg-gradient-to-br from-[#d8ebe4] via-[#edf5ef] to-[#f7fbfb] p-8 shadow-[0_30px_100px_rgba(118,176,165,0.26)]">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div className="max-w-2xl">
              <p className="text-xs uppercase tracking-[0.28em] text-neutral-500">Organization Control Room</p>
              <h2 className="mt-3 text-4xl font-semibold tracking-tight text-neutral-900">
                Review seeded applications and new session submissions together
              </h2>
              <p className="mt-4 text-base leading-7 text-neutral-600">
                This workspace keeps historical sample applicants visible for analyst review while also
                surfacing any new applications created through the customer flow.
              </p>
            </div>
            <div className="grid gap-3 sm:grid-cols-2">
              <OrgChip icon={<Users className="h-4 w-4" />} label={`${applications.filter((a) => a.source === 'seed').length} seeded records`} />
              <OrgChip icon={<BriefcaseBusiness className="h-4 w-4" />} label={`${applications.filter((a) => a.source === 'customer').length} customer session records`} />
              <OrgChip icon={<Activity className="h-4 w-4" />} label="Analyst workflow ready" />
              <OrgChip icon={<Clock3 className="h-4 w-4" />} label="Human review queue visible" />
            </div>
          </div>
        </div>

        <div className="grid gap-4">
          <Card className="rounded-[30px] border-white/80 bg-neutral-900 text-white">
            <div className="space-y-3">
              <p className="text-xs uppercase tracking-[0.2em] text-white/60">Data Sources</p>
              <p className="text-4xl font-semibold">{applications.length}</p>
              <div className="grid gap-3 text-sm text-white/75">
                <div className="flex items-center justify-between">
                  <span>Seeded Org Applicants</span>
                  <span>{applications.filter((a) => a.source === 'seed').length}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span>Live Customer Applications</span>
                  <span>{applications.filter((a) => a.source === 'customer').length}</span>
                </div>
              </div>
            </div>
          </Card>
        </div>
      </section>

      {metrics && <MetricsGrid metrics={metrics} />}
      {metrics && (
        <section className="mb-8 grid grid-cols-1 gap-6 md:grid-cols-3">
          <KPICard label="Avg Loan Amount" value={metrics.avgLoanAmount} format="currency" />
          <KPICard label="Approval Rate" value={metrics.approvalRate} format="percentage" />
          <KPICard label="Automation Rate" value={metrics.automationRate} format="percentage" />
        </section>
      )}

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
              <LineChart data={trends}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="count" stroke="#16a34a" strokeWidth={3} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card title="Category Analysis">
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={categoryAnalysis}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="label" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="value" fill="#0ea5e9" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card title="Risk Score Distribution">
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={riskScoreDistribution}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="label" />
                <YAxis />
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
          <h3 className="text-lg font-semibold text-neutral-900 mb-6">Approval Distribution</h3>
          <div>
            <p className="text-sm text-neutral-500">Average Loan Amount</p>
            <p className="text-2xl font-semibold text-neutral-900">
              {metrics ? formatCurrency(metrics.avgLoanAmount) : '--'}
            </p>
          </div>
          <div>
            <p className="text-sm text-neutral-500">Average Processing Time</p>
            <p className="text-2xl font-semibold text-neutral-900">
              {metrics ? `${Math.round(metrics.averageProcessingTime / 60)} min` : '--'}
            </p>
          </div>
          <div>
            <p className="text-sm text-neutral-500">Human Review Queue</p>
            <p className="text-2xl font-semibold text-neutral-900">
              {applications.filter((application) => application.status === 'deferred').length}
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
            Deferred Review ({metrics?.deferred ?? 0})
          </button>
        </div>

        <div className="p-6">
          <ApplicationTable
            data={displayApplications}
            onRowClick={handleRowClick}
            isLoading={false}
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
