import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Activity, BrainCircuit, BriefcaseBusiness, CheckCircle2, MapPinned,
  XCircle, AlertTriangle,
} from 'lucide-react'
import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, Cell,
  Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import { DashboardLayout } from '@/components/layouts/DashboardLayout'
import { Button, Card, KPICard } from '@/components/common'
import { ApplicationTable } from '@/components/sections'
import { useApplicationData } from '@/hooks/useApplicationData'
import { getStats } from '@/services/applications'
import type { StatsResponse } from '@/types/api'
import { formatCurrency } from '@/lib/utils'
import { SERIES, useChartTheme } from '@/lib/chartTheme'
import type { LoanApplication } from '@/types/application'

type OrgTab = 'all' | 'deferred' | 'approved' | 'rejected' | 'confirmed'

const TAB_CONFIG: Array<{ id: OrgTab; label: string; color: string; activeColor: string }> = [
  { id: 'all', label: 'All Applications', color: 'text-neutral-600', activeColor: 'border-primary-600 text-primary-700 bg-primary-50' },
  { id: 'deferred', label: 'Needs Review', color: 'text-amber-600', activeColor: 'border-amber-500 text-amber-700 bg-amber-50' },
  { id: 'approved', label: 'Auto-Approved', color: 'text-green-600', activeColor: 'border-green-500 text-green-700 bg-green-50' },
  { id: 'rejected', label: 'Auto-Rejected', color: 'text-red-600', activeColor: 'border-red-500 text-red-700 bg-red-50' },
  { id: 'confirmed', label: 'Org-Confirmed', color: 'text-violet-600', activeColor: 'border-violet-500 text-violet-700 bg-violet-50' },
]

export const OrganizationDashboard: React.FC = () => {
  const navigate = useNavigate()
  const { applications, isLoading, error, bulkOverrideDecision } = useApplicationData({ scope: 'org' })
  const [activeTab, setActiveTab] = useState<OrgTab>('all')
  const [stats, setStats] = useState<StatsResponse | null>(null)
  const [dashboardError, setDashboardError] = useState<string | null>(null)
  const [statsLoading, setStatsLoading] = useState(true)
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [bulkNotes, setBulkNotes] = useState('')
  const [bulkLoading, setBulkLoading] = useState(false)
  const [bulkError, setBulkError] = useState<string | null>(null)
  const [showBulkPanel, setShowBulkPanel] = useState(false)
  const chart = useChartTheme()

  useEffect(() => {
    const load = async () => {
      setStatsLoading(true)
      try {
        const s = await getStats()
        setStats(s)
      } catch (e) {
        setDashboardError(e instanceof Error ? e.message : 'Failed to load stats')
      } finally {
        setStatsLoading(false)
      }
    }
    void load()
  }, [])

  // Filter logic for tabs
  const tabFiltered = useMemo<LoanApplication[]>(() => {
    if (activeTab === 'all') return applications
    if (activeTab === 'deferred') return applications.filter((a) => a.modelRecommendation === 'deferred' && !a.manualDecisionApplied)
    if (activeTab === 'approved') return applications.filter((a) => a.modelRecommendation === 'approved' && !a.manualDecisionApplied)
    if (activeTab === 'rejected') return applications.filter((a) => a.modelRecommendation === 'rejected' && !a.manualDecisionApplied)
    if (activeTab === 'confirmed') return applications.filter((a) => a.manualDecisionApplied)
    return applications
  }, [applications, activeTab])

  const tabCounts = useMemo(() => ({
    all: applications.length,
    deferred: applications.filter((a) => a.modelRecommendation === 'deferred' && !a.manualDecisionApplied).length,
    approved: applications.filter((a) => a.modelRecommendation === 'approved' && !a.manualDecisionApplied).length,
    rejected: applications.filter((a) => a.modelRecommendation === 'rejected' && !a.manualDecisionApplied).length,
    confirmed: applications.filter((a) => a.manualDecisionApplied).length,
  }), [applications])

  const uploadedCount = applications.filter((a) => a.source === 'seed').length
  const submittedCount = applications.filter((a) => a.source === 'customer').length

  const handleRowClick = (app: { id: string }) => navigate(`/review/${app.id}`)

  const toggleSelect = useCallback((id: string) => {
    setSelected((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }, [])

  const toggleSelectAll = () => {
    if (selected.size === tabFiltered.length) {
      setSelected(new Set())
    } else {
      setSelected(new Set(tabFiltered.map((a) => a.id)))
    }
  }

  const handleBulkAction = async (status: 'approved' | 'rejected') => {
    if (!bulkNotes.trim()) {
      setBulkError('Please provide notes for the bulk decision.')
      return
    }
    if (selected.size === 0) {
      setBulkError('No applications selected.')
      return
    }
    setBulkError(null)
    setBulkLoading(true)
    try {
      await bulkOverrideDecision(Array.from(selected), status, bulkNotes)
      setSelected(new Set())
      setBulkNotes('')
      setShowBulkPanel(false)
    } catch (e) {
      setBulkError(e instanceof Error ? e.message : 'Bulk action failed')
    } finally {
      setBulkLoading(false)
    }
  }

  const trends = useMemo(() => {
    const now = new Date()
    const weeks = [
      { label: 'Week 1', start: new Date(now.getTime() - 28 * 86400000), end: new Date(now.getTime() - 21 * 86400000) },
      { label: 'Week 2', start: new Date(now.getTime() - 21 * 86400000), end: new Date(now.getTime() - 14 * 86400000) },
      { label: 'Week 3', start: new Date(now.getTime() - 14 * 86400000), end: new Date(now.getTime() - 7 * 86400000) },
      { label: 'Week 4', start: new Date(now.getTime() - 7 * 86400000), end: new Date(now.getTime() + 86400000) },
    ]
    return weeks.map((w) => {
      const bucket = applications.filter((a) => {
        const d = new Date(a.createdAt)
        return d >= w.start && d < w.end
      })
      return {
        date: w.label,
        count: bucket.length,
        approved: bucket.filter((a) => a.finalDecision === 'APPROVE').length,
        rejected: bucket.filter((a) => a.finalDecision === 'REJECT').length,
        deferred: bucket.filter((a) => a.finalDecision === 'DEFER').length,
      }
    })
  }, [applications])

  const approvalDistribution = useMemo(() => [
    { label: 'Approved', value: stats?.approved ?? 0, fill: SERIES.approve },
    { label: 'Rejected', value: stats?.rejected ?? 0, fill: SERIES.reject },
    { label: 'Deferred', value: stats?.deferred ?? 0, fill: SERIES.defer },
  ], [stats])

  const categoryAnalysis = useMemo(() => {
    const counts = new Map<string, number>()
    for (const a of applications) counts.set(a.loanPurpose, (counts.get(a.loanPurpose) ?? 0) + 1)
    return Array.from(counts.entries()).map(([label, value]) => ({
      label: label.charAt(0).toUpperCase() + label.slice(1),
      value,
    }))
  }, [applications])

  const decisionTotal = approvalDistribution.reduce((sum, item) => sum + item.value, 0)
  const avgLoanAmount = applications.length
    ? formatCurrency(Math.round(applications.reduce((sum, a) => sum + a.loanAmount, 0) / applications.length))
    : '--'
  const statValue = (value: number | undefined) => (statsLoading ? '…' : value ?? '—')

  return (
    <DashboardLayout title="Dashboard" role="organization">
      {/* Page header */}
      <section className="mb-6 flex flex-wrap items-end justify-between gap-4">
        <div className="max-w-2xl">
          <h1 className="text-2xl font-semibold tracking-tight text-neutral-900 sm:text-3xl">Operations overview</h1>
          <p className="mt-1 text-sm leading-6 text-neutral-500">
            All records in one queue. Review, override, and confirm ML decisions. Deferred cases require human action before the customer is notified.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button variant="secondary" size="sm" leftIcon={<BrainCircuit className="h-4 w-4" />} onClick={() => navigate('/dashboard/models')}>
            Model Analysis
          </Button>
          <Button variant="primary" size="sm" leftIcon={<MapPinned className="h-4 w-4" />} onClick={() => navigate('/analytics/geo')}>
            Geo Analytics
          </Button>
        </div>
      </section>

      {(error || dashboardError) && (
        <section className="mb-6">
          <Card className="border-red-200 bg-red-50">
            <p className="text-red-700">Connection issue: {error ?? dashboardError}</p>
          </Card>
        </section>
      )}

      {/* KPI row */}
      <section className="mb-6 grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <KPICard
          label="Total Applications"
          value={statValue(stats?.totalApplications)}
          icon={BriefcaseBusiness}
          tone="blue"
          hint={`${submittedCount} live · ${uploadedCount} training records`}
        />
        <KPICard
          label="Approval Rate"
          value={statsLoading ? '…' : stats ? `${stats.approvalRate}%` : '—'}
          icon={CheckCircle2}
          tone="green"
          hint={stats ? `Rejection rate ${stats.rejectionRate}%` : undefined}
        />
        <KPICard
          label="Needs Review"
          value={tabCounts.deferred}
          icon={AlertTriangle}
          tone="amber"
          hint={stats ? `Deferral rate ${stats.deferralRate}%` : undefined}
        />
        <KPICard
          label="Average CBES"
          value={statValue(stats?.averageCBES)}
          icon={Activity}
          tone="violet"
          hint={stats ? `Average ML score ${stats.averageMLProbability}` : undefined}
        />
      </section>

      {/* Trend + decision mix */}
      <section className="mb-6 grid grid-cols-1 gap-6 lg:grid-cols-3">
        <Card title="Applications Over Time" description="Final decisions per week, last 4 weeks" className="lg:col-span-2">
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={trends} margin={{ top: 12, right: 12, bottom: 0, left: -12 }}>
                <defs>
                  {(['approve', 'reject', 'defer'] as const).map((key) => (
                    <linearGradient key={key} id={`org-${key}`} x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor={SERIES[key]} stopOpacity={0.25} />
                      <stop offset="100%" stopColor={SERIES[key]} stopOpacity={0} />
                    </linearGradient>
                  ))}
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke={chart.grid} vertical={false} />
                <XAxis dataKey="date" {...chart.axisProps} />
                <YAxis allowDecimals={false} width={44} {...chart.axisProps} />
                <Tooltip {...chart.tooltipProps} />
                <Area type="monotone" dataKey="approved" name="Approved" stroke={SERIES.approve} strokeWidth={2} fill="url(#org-approve)" />
                <Area type="monotone" dataKey="rejected" name="Rejected" stroke={SERIES.reject} strokeWidth={2} fill="url(#org-reject)" />
                <Area type="monotone" dataKey="deferred" name="Deferred" stroke={SERIES.defer} strokeWidth={2} fill="url(#org-defer)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card title="Decision Mix" description="Across all applications">
          <div className="relative h-56">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={approvalDistribution}
                  dataKey="value"
                  nameKey="label"
                  innerRadius={62}
                  outerRadius={90}
                  paddingAngle={2}
                  stroke={chart.surface}
                  strokeWidth={2}
                >
                  {approvalDistribution.map((e) => <Cell key={e.label} fill={e.fill} />)}
                </Pie>
                <Tooltip {...chart.tooltipProps} />
              </PieChart>
            </ResponsiveContainer>
            <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center">
              <span className="text-3xl font-semibold text-neutral-900">{decisionTotal}</span>
              <span className="text-xs text-neutral-500">Decisions</span>
            </div>
          </div>
          <div className="mt-2 grid grid-cols-3 gap-2 border-t border-neutral-200 pt-4 text-center">
            {approvalDistribution.map((item) => (
              <div key={item.label}>
                <p className="flex items-center justify-center gap-1.5 text-xs text-neutral-500">
                  <span className="h-2 w-2 rounded-full" style={{ background: item.fill }} />
                  {item.label}
                </p>
                <p className="mt-1 text-lg font-semibold text-neutral-900">{item.value}</p>
              </div>
            ))}
          </div>
        </Card>
      </section>

      {/* Purpose mix + pipeline */}
      <section className="mb-6 grid grid-cols-1 gap-6 lg:grid-cols-3">
        <Card title="Loan Purpose Mix" description="Applications by stated purpose" className="lg:col-span-2">
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={categoryAnalysis} margin={{ top: 12, right: 12, bottom: 28, left: -12 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={chart.grid} vertical={false} />
                <XAxis dataKey="label" interval={0} angle={-18} textAnchor="end" height={56} {...chart.axisProps} />
                <YAxis allowDecimals={false} width={44} {...chart.axisProps} />
                <Tooltip {...chart.tooltipProps} />
                <Bar dataKey="value" name="Applications" fill={SERIES.brand} radius={[6, 6, 0, 0]} maxBarSize={48} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card title="Application Pipeline" description="Where every application stands">
          <p className="text-4xl font-semibold tracking-tight text-neutral-900">{applications.length}</p>
          <div className="mt-4 space-y-1">
            <PipelineRow icon={<CheckCircle2 className="h-4 w-4 text-green-600" />} label="Auto-Approved" value={tabCounts.approved} />
            <PipelineRow icon={<XCircle className="h-4 w-4 text-red-600" />} label="Auto-Rejected" value={tabCounts.rejected} />
            <PipelineRow icon={<AlertTriangle className="h-4 w-4 text-amber-600" />} label="Needs Review" value={tabCounts.deferred} />
            <PipelineRow icon={<CheckCircle2 className="h-4 w-4 text-violet-600" />} label="Org-Confirmed" value={tabCounts.confirmed} />
          </div>
          <div className="mt-4 flex items-center justify-between rounded-xl bg-neutral-100 px-4 py-3">
            <span className="text-sm text-neutral-500">Avg loan amount</span>
            <span className="font-semibold text-neutral-900">{avgLoanAmount}</span>
          </div>
        </Card>
      </section>

      {/* Application Table with Tabs + Bulk Actions */}
      <div className="overflow-hidden rounded-2xl border border-neutral-200 bg-white shadow-[0_1px_2px_rgba(15,23,42,0.04)]">
        {/* Tab Bar */}
        <div className="border-b border-neutral-200 flex overflow-x-auto">
          {TAB_CONFIG.map((tab) => (
            <button
              key={tab.id}
              onClick={() => { setActiveTab(tab.id); setSelected(new Set()) }}
              className={`flex-shrink-0 px-5 py-4 text-sm font-medium border-b-2 transition-colors ${
                activeTab === tab.id
                  ? `border-b-2 ${tab.activeColor}`
                  : `border-transparent ${tab.color} hover:bg-neutral-50`
              }`}
            >
              {tab.label} ({tabCounts[tab.id]})
            </button>
          ))}
        </div>

        {/* Bulk Action Toolbar */}
        {(activeTab === 'approved' || activeTab === 'rejected' || activeTab === 'deferred') && (
          <div className="border-b border-neutral-100 bg-neutral-50 px-6 py-3 flex flex-wrap items-center gap-4">
            <label className="flex items-center gap-2 text-sm text-neutral-600 cursor-pointer">
              <input
                type="checkbox"
                className="w-4 h-4 rounded accent-primary-600"
                checked={selected.size > 0 && selected.size === tabFiltered.length}
                onChange={toggleSelectAll}
              />
              {selected.size > 0 ? `${selected.size} selected` : 'Select all'}
            </label>
            {selected.size > 0 && (
              <button
                onClick={() => setShowBulkPanel(!showBulkPanel)}
                className="text-sm font-medium text-primary-600 hover:underline"
              >
                {showBulkPanel ? 'Hide bulk panel' : 'Bulk action →'}
              </button>
            )}
          </div>
        )}

        {/* Bulk Panel */}
        {showBulkPanel && selected.size > 0 && (
          <div className="bg-primary-50 border-b border-primary-100 px-6 py-4 flex flex-wrap gap-4 items-start">
            <div className="flex-1 min-w-[240px]">
              <label className="block text-xs font-semibold text-primary-900 mb-1">Decision Notes (required)</label>
              <input
                type="text"
                value={bulkNotes}
                onChange={(e) => setBulkNotes(e.target.value)}
                placeholder="e.g. Batch confirmed after committee review"
                className="w-full rounded-lg border border-primary-200 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-400"
              />
              {bulkError && <p className="mt-1 text-xs text-red-600">{bulkError}</p>}
            </div>
            <div className="flex gap-2 pt-5">
              <button
                onClick={() => void handleBulkAction('approved')}
                disabled={bulkLoading}
                className="flex items-center gap-1.5 rounded-lg bg-green-600 px-4 py-2 text-sm font-semibold text-white hover:bg-green-700 disabled:opacity-50"
              >
                <CheckCircle2 className="h-4 w-4" />
                {bulkLoading ? 'Processing…' : `Approve ${selected.size}`}
              </button>
              <button
                onClick={() => void handleBulkAction('rejected')}
                disabled={bulkLoading}
                className="flex items-center gap-1.5 rounded-lg bg-red-600 px-4 py-2 text-sm font-semibold text-white hover:bg-red-700 disabled:opacity-50"
              >
                <XCircle className="h-4 w-4" />
                {bulkLoading ? 'Processing…' : `Reject ${selected.size}`}
              </button>
            </div>
          </div>
        )}

        {/* Table */}
        <div className="p-6">
          {activeTab === 'deferred' && tabCounts.deferred > 0 && (
            <div className="mb-4 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800 flex items-center gap-2">
              <AlertTriangle className="h-4 w-4 shrink-0" />
              These {tabCounts.deferred} cases require an analyst decision. Customers are in a pending state until action is taken.
            </div>
          )}
          <ApplicationTable
            data={tabFiltered}
            onRowClick={handleRowClick}
            isLoading={isLoading}
            pageSize={25}
            showApplicant
            selectedIds={selected}
            onToggleSelect={
              (activeTab === 'approved' || activeTab === 'rejected' || activeTab === 'deferred')
                ? toggleSelect
                : undefined
            }
          />
        </div>
      </div>
    </DashboardLayout>
  )
}

function PipelineRow({ icon, label, value }: { icon: React.ReactNode; label: string; value: number }) {
  return (
    <div className="flex items-center justify-between border-b border-neutral-200 py-2.5 text-sm last:border-0">
      <span className="flex items-center gap-2 text-neutral-600">{icon}{label}</span>
      <span className="font-semibold text-neutral-900">{value}</span>
    </div>
  )
}
